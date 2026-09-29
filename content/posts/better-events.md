+++
date = '2026-09-24T14:24:35+01:00'
title = 'Event-based data: a better way to represent events'
draft = true
+++

When working with event-based data, changing the event memory layout can save up to 70% memory while making event-based algorithms faster.

> Disclaimer: This article is written by a human, however, LLMs have been used to code and run the benchmarks, generate tables and figures in this post.

## Intro 

Event cameras generate an asynchronous stream of events: basically, what you get out of the camera is a list of events, where one event consists of x,y coordinates, a timestamp and a polarity (boolean ON or OFF).

In Prophesee's [OpenEB library](https://github.com/prophesee-ai/openeb/blob/9003b5416676e78ba994d912087486cfa94fae73/sdk/modules/base/cpp/include/metavision/sdk/base/events/event2d.h#L30), an event is stored as:

```cpp
struct Event {
    uint16_t x;
    uint16_t y;
    int16_t p;
    int64_t t; // timestamp in microseconds
}
```

And the API gives us a `std::vector<Event>` in a callback.


Already we can see from the layout that it's not great:

```
offset  size
0       2     uint16_t x
2       2     uint16_t y
4       2     int16_t  p
6       2     padding
8       8     int64_t t
------------------------
total  16 bytes
```

One event is 16 bytes, with 2 bytes used just for padding! That's **128 bits per event**! An event camera can generate millions of events per second, so the memory footprint of a chunk of events can be huge and hurt the performance of algorithms.

Another library, EventCV, uses a [structure of array (SoA)](https://github.com/EventLAB-Team/eventcv/blob/5cc1ae1b404de6f397b9fa72399be5a3925bb4be/crates/eventcv-core/src/lib.rs#L74) where x, y, p, and t each live in their own array. That eliminates padding so the footprint is 13 bytes per event. That's an improvement, but we can do way better!

## How can it be improved?


### Pack stuff

Packing will save us a lot of memory, and it will come at the cost of unpacking, but let's give it a go:

Let's say our format supports sensors up to 2048x2048. That's a limitation, but it also means we can use only 11 bits for each coordinate, and because we need one bit for the polarity, the proposed layout is:

```
bit: 31                      23 22 21          11 10               0
     +-------------------------+--+---------------+----------------+
     |       unused (9)        |P |     Y (11)    |    X (11)      |
     +-------------------------+--+---------------+----------------+
```

This way, a single event x, y, p fits comfortably in a single register. We also have 9 free bits that we can use to store information or metadata about an event, like an external trigger or something else.

I can already hear you scream: "But it's terrible, you will have to unpack these every time!!". The key idea: You don't! You don't have to unpack it if you treat this as a single number that can index into an array. You have 2 choices: either use the full 23-bit value as an index into a 3D array [p][y][x], or mask the polarity bit and use the 22-bit integer to index into a row-major [y][x] array. Both of those arrays will have a large **stride** (i.e. empty space at the end of rows), but we'll discuss that later. 

Let's call this value the `EventID` (for event index). It doesn't depend on time; it identifies an event's coordinates and polarity. Let's say you want to build a surface of active events (which is just a map of the most recent activity).


```rust
const W: usize = 1280;           // sensor width
const H: usize = 720;            // sensor height
const STRIDE: usize = 2048;      // the EventId's own row stride

// Surface of active events (SAE), stores the last timestamp for each pixel.
// Here we ignore polarity, but we could have a 2-channel SAE, one for each polarity
let mut sae: Vec<u32> = vec![0; H * STRIDE];

// Write the SAE. Never allocates, never blocks, never reads.
loop {
    for (id, ts) in camera.next_event() { // Note: this is a fake API, we'll see later how to iterate over events
        // The id packs polarity, y and x. Masking off the polarity bit
        // leaves y * 2048 + x -- already a row-major pixel index, so the
        // event IS the address. No unpacking, no y * width + x.
        sae[id & PIXEL_MASK] = ts;
    }
}
```


So basically, by storing both X and Y row-major on 11 bits each, we are using a stride of 2¹¹, i.e. 2048. This is great, but we can actually adapt this stride according to our needs. For instance, if we want to index directly into an array we can use y*width+x. In this case our layout becomes:

```
 Adapted stride (=width)  id & PIXEL_MASK  ==  y * width + x

  31            23 22 21                             0
 ┌────────────────┬──┬────────────────────────────────┐
 │      free      │p │          y * width + x         │
 └────────────────┴──┴────────────────────────────────┘
                      └───────────────┬───────────────┘
                                      └─ 22 bits, no field boundary
```

If we want to do branchless neighborhood checks, for instance for a background activity filter that looks at the 8 neighbors of each event, we can use STRIDE=width+2 to add two more columns and two more rows to our arrays. This way, an algorithm that checks the neighbors doesn't have to do `if (x>WIDTH-1) etc...`; instead, we can efficiently do: 

```rust
  const S: i32 = WIDTH as i32 + 2;               // S is the Stride, with one guard column on each side
  let p = (y as i32 + 1) * S + (x as i32 + 1);   // pixel index, offset +1 row, +1 column for the guards
  let t = ts + DT;                               // biased clock so that empty cells (0) never match
  let mut hit = false;
  for o in [-1, 1, -S, S, -S - 1, -S + 1, S - 1, S + 1] {
      hit |= t.wrapping_sub(sae[(p + o) as usize]) < DT;
  }
  sae[p as usize] = t;
```


The diagrams below try to represent those different buffer layouts depending on the stride:


![Architecture diagram](/better-events-diagrams.svg)


#### The EVT3 wire format bonus! 

When the data comes out of the camera, it's actually encoded in a more compact representation, usually EVT2.1 or EVT3. The first step is to decode this data.

EVT3 uses 16-bit words where, sometimes, X and Y are already encoded on 11 bits, so packing an event can become a single `OR`.

I don't want to go into too much detail about the decoder in this article, but the main idea is that EVT3 already gives this 2048 upper bound for the sensor resolution. 


#### Wrapping up

So finally, the decoded event representation is encoded in a 32-bit integer as: 

`bit 22 = polarity | bits 21:0 = pixel = y * STRIDE + x`


The final struct is just a wrapper around a `u32`:

```rust 
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct EventId<const STRIDE: u32 = 2048>(u32); // Use 2048 as default stride
```

Here's an example implementation; I omitted `#[inline(always)]` and tried to keep it simple.

```rust 
impl<const STRIDE: u32> EventId<STRIDE> {
    pub const STRIDE: u32 = STRIDE;
    pub const POL_BIT: u32 = 1 << 22;
    pub const PIXEL_MASK: u32 = (1 << 22) - 1;

    pub const fn new(x: u32, y: u32, pol: bool) -> Self {
        let pixel = y * STRIDE + x;
        debug_assert!(pixel <= Self::PIXEL_MASK);
        Self(pixel | if pol { Self::POL_BIT } else { 0 })
    }

    pub const fn from_u32(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn to_u32(self) -> u32 {
        self.0
    }

    /// Row-major pixel index, polarity stripped: y * STRIDE + x.
    pub const fn pixel(self) -> u32 {
        self.0 & Self::PIXEL_MASK
    }

    /// Index into a polarity-split array [pol][y][x]: pol * 2^22 + y * STRIDE + x.
    /// The polarity stride is fixed at 2^22, so the array needs 2^23 cells.
    pub const fn pol_pixel(self) -> u32 {
        self.0 & (Self::POL_BIT | Self::PIXEL_MASK)
    }

    pub const fn pol(self) -> bool {
        self.0 & Self::POL_BIT != 0
    }

    pub const fn x(self) -> u32 {
        self.pixel() % STRIDE
    }

    pub const fn y(self) -> u32 {
        self.pixel() / STRIDE
    }
}
```

### Picking a stride

By making STRIDE a const generic, defaulting to 2048, we can choose the stride at compile time,
and if we look at the generated x86 assembly, unpacking looks like this:


**Stride 2048:** 

```asm
x_2048:  and eax, 2047        ; x = low 11 bits (this also drops the polarity bit)
y_2048:  shr eax, 11          ; move y down to the low bits
         and eax, 2047        ; keep 11 bits (drops polarity)
```

**Stride 1280:**

```asm
y_1280:  and  edi, 4194048        ; drop polarity (and the low 8 bits, which can't change the result)
         imul rax, rdi, 3355444   ; multiply by ≈ 2^32 / 1280
         shr  rax, 32             ; keep the high half: this IS v / 1280

x_1280:  and  eax, 4194303        ; drop polarity
         imul rcx, rax, 3355444   ; y = v / 1280, same trick as above
         shr  rcx, 32
         shl  ecx, 8              ; y * 256
         lea  ecx, [rcx + 4*rcx]  ; ... * 5 = y * 1280
         sub  eax, ecx            ; x = v - y * 1280
```
> (check this on [Godbolt](https://godbolt.org/#g:!((g:!((g:!((h:codeEditor,i:(filename:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,selection:(endColumn:101,endLineNumber:24,positionColumn:101,positionLineNumber:24,selectionStartColumn:101,selectionStartLineNumber:24,startColumn:101,startLineNumber:24),source:'%23%5Bderive(Clone,+Copy)%5D%0Apub+struct+EventId%3Cconst+STRIDE:+u32+%3D+2048%3E(u32)%3B%0A%0Aimpl%3Cconst+STRIDE:+u32%3E+EventId%3CSTRIDE%3E+%7B%0A++++pub+const+PIXEL_MASK:+u32+%3D+(1+%3C%3C+22)+-+1%3B%0A%0A++++pub+const+fn+from_u32(raw:+u32)+-%3E+Self+%7B+Self(raw)+%7D%0A%0A++++///+Row-major+pixel+index,+polarity+stripped.%0A++++pub+const+fn+index(self)+-%3E+u32+%7B+self.0+%26+Self::PIXEL_MASK+%7D%0A%0A++++pub+const+fn+x(self)+-%3E+u32+%7B+self.index()+%25+STRIDE+%7D%0A++++pub+const+fn+y(self)+-%3E+u32+%7B+self.index()+/+STRIDE+%7D%0A%7D%0A%0A//+Stride+known+at+compile+time+(const+generic).%0A%23%5Binline(never)%5D+pub+fn+x_2048(raw:+u32)+-%3E+u32+%7B+EventId::%3C2048%3E::from_u32(raw).x()+%7D%0A%23%5Binline(never)%5D+pub+fn+y_2048(raw:+u32)+-%3E+u32+%7B+EventId::%3C2048%3E::from_u32(raw).y()+%7D%0A%23%5Binline(never)%5D+pub+fn+x_1280(raw:+u32)+-%3E+u32+%7B+EventId::%3C1280%3E::from_u32(raw).x()+%7D%0A%23%5Binline(never)%5D+pub+fn+y_1280(raw:+u32)+-%3E+u32+%7B+EventId::%3C1280%3E::from_u32(raw).y()+%7D%0A%0A//+For+contrast:+stride+only+known+at+runtime.%0A%23%5Binline(never)%5D+pub+fn+x_runtime(raw:+u32,+stride:+u32)+-%3E+u32+%7B+(raw+%26+((1+%3C%3C+22)+-+1))+%25+stride+%7D%0A%23%5Binline(never)%5D+pub+fn+y_runtime(raw:+u32,+stride:+u32)+-%3E+u32+%7B+(raw+%26+((1+%3C%3C+22)+-+1))+/+stride+%7D'),l:'5',n:'0',o:'Rust+source+%231',t:'0')),k:42.70352760799251,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:compiler,i:(compiler:r1980,filters:(b:'0',binary:'1',binaryObject:'1',commentOnly:'0',debugCalls:'1',demangle:'0',directives:'0',execute:'1',intel:'0',libraryCode:'0',trim:'1',verboseDemangling:'0'),flagsViewOpen:'1',fontScale:14,fontUsePx:'0',j:1,lang:rust,libs:!(),options:'-C+opt-level%3D3',overrides:!((name:edition,value:'2024')),selection:(endColumn:1,endLineNumber:1,positionColumn:1,positionLineNumber:1,selectionStartColumn:1,selectionStartLineNumber:1,startColumn:1,startLineNumber:1),source:1),l:'5',n:'0',o:'+rustc+1.98.0+(Editor+%231)',t:'0')),k:33.78132083110647,l:'4',n:'0',o:'',s:0,t:'0'),(g:!((h:output,i:(compilerName:'x86-64+gcc+15.2',editorid:1,fontScale:14,fontUsePx:'0',j:1,wrap:'1'),l:'5',n:'0',o:'Output+of+rustc+1.98.0+(Compiler+%231)',t:'0')),k:23.515151560901,l:'4',n:'0',o:'',s:0,t:'0')),l:'2',n:'0',o:'',t:'0')),version:4))

Unpacking x and y is obviously more costly when the stride is not a power of two, but the whole point of this layout is to never unpack and use the EventId directly to index memory.

## What about timestamps?


The original OpenEB `Event` stores the timestamp as an i64 microsecond count. 63 bits is 
9223372036854775808 microseconds; that's more than 292 000 years!! It's extremely wasteful to store this many bits for every event.

I suggest using only 32 bits for the timestamp, which gives us around 71 minutes. That's not much, but we can always store an absolute time offset alongside our recordings, and use chunks of 71 minutes.

With our u32 packing x,y,p and u32 timestamp, we are now using 8 bytes per event! We halved the memory used by OpenEB, but we can do better!

## Come on! Don't store all timestamps

Multiple events can share the same timestamp. Why bother storing the same timestamp everywhere? What we could do instead is *store it only when the timestamp changes*.

The idea is to define a `TimeMark` such that: events are sorted by timestamp, and each TimeMark points to the first event of a contiguous timestamp segment.

```
struct TimeMark {
    ts_us: u32,       // timestamp, µs relative to the buffer origin
    event_idx: u32,   // index of the first event with this timestamp
}
```

And we store event buffers as event ids (x,y,p) and time marks; that's simply a Run-Length Encoding (RLE) format.

```
struct EventBuf {
    ids:   Vec<EventId>,    // 4 Bytes per event
    marks: Vec<TimeMark>,   // 8 Bytes per *distinct timestamp*
    t_origin: u64,          // absolute start, stored once
}
```

For this to work, we need to have long *segments* of events with the same timestamp. This is very data-dependent, but on a very small test on a single recording we can see: 


todo(add svg)

Which gives: 

| Layout | Bytes/event |
|---|---:|
| vector of OpenEB `Event` | 16.00 |
| event_id + `u64` per event (OpenEB-ish) | 12.00 |
| event_id + `u32` per event | 8.00 |
| **id + 8 B marks** | 4.75 |


Which means we can go from 16 Bytes per event to 4.75 Bytes per event! **The new representation is a 70% memory reduction versus the original `vector<Event>` in OpenEB!**

If we compare it to the baseline EVT3 wire format, we can see:

| Representation | Size | Bytes/event | vs raw EVT3 |
|---|---|---|---|
| `vector<Event>` | 1,783.8 MB | 16.00 | 4.86x |
| `EventId` + `TimeMarks` | 530.0 MB | 4.75 | 1.44x |
| Raw EVT3 | 367.3 MB | 3.29 | 1.00x |

We use only 1.44x more memory while being fully decoded and so much more convenient for algorithms.

## Small recap

```
// A single u32 that packs polarity, x and y
// this single number can be used to index directly into an array without unpacking
struct EventID(u32); 
    
// A timestamp mark, used to define the start of a segment where events share the same timestamp
// It's just the timestamp of the first event and the index at which it appears
struct TimeMark {
    ts_us: u32,       // timestamp, µs relative to the buffer origin
    event_idx: u32,   // index of the first event with this timestamp
}


// A buffer of events is a list of ids, time marks, and a time offset. 
struct EventBuf {
    ids:   Vec<EventId>,    // 4 Bytes per event
    marks: Vec<TimeMark>,   // 8 Bytes per *distinct timestamp*
    t_origin: u64,          // absolute start, stored once
}
```

## What about the consumers?

I hear you cry: "But man! Now I have to unpack my EventIDs to extract x,y,p and I have to deal with weird TimeMarks".

I've already tried to show that you don't necessarily need to unpack the EventIDs, and the goal of this single `u32` is to use it to index directly into 1D arrays.


For the timestamp, the `TimeMark` design allows us to iterate over **segments**. This is very useful and can be **more efficient** for certain processing. Here is an example of how to compute a time-binned histogram (voxel):


```rust
const BINS: usize = 10;
const PLANE: usize = H * STRIDE;          // one time bin = one image

let mut voxels = vec![0i32; BINS * PLANE]; // output voxel
let t0 = buf.first_ts();
let span = (buf.last_ts() + 1 - t0) as usize;

for (ts, ids) in buf.segments() { // a segment returns one timestamp (ts) and a slice of EventIds (ids).
    // Every event in a segment shares `ts`, so the bin is computed
    // once per segment (~11 events on average), not once per event.
    let bin = (ts - t0) as usize * BINS / span;
    let plane = &mut voxels[bin * PLANE ..][..PLANE];

    for &id in ids {
        // No timestamp read, no unpacking: the id is the address.
        plane[id.pixel() as usize] += if id.pol() { 1 } else { -1 };
    }
}
```


## Benchmarks

I ran a few benchmarks on a single recording just to show that this API makes sense. A more complete benchmark on more data is definitely needed to draw conclusions.


I compared OpenEB's array of structures, EventCV's structure of arrays, and this EventBuf implementation:

Size of the container for the whole recording:

| Layout | Bytes/event | Total |
|---|---:|---:|
| OpenEB `Vec<Event>` | 16.00 | 1.78 GB |
| EventCV SoA `EventStream` | 13.00 | 1.45 GB |
| our `EventBuf` (`EventId` + `TimeMark`) | 4.75 | 530 MB |


Then I ran some classic event-based algorithms, all with the same Rust runtime but changing only the event representation, to see whether this new representation has an impact.

Times are total milliseconds over 417 slices of 30 ms, on a single core of a Ryzen 7 PRO 7840U, with every implementation reusing its buffers and producing output bit-identical to EventCV's. EventBuf is fastest in five of six algorithms, while using 3.4x less memory than Vec<Event> and 2.7x less than EventCV.


| Algorithm | `EventBuf` | `Vec<Event>` | EventCV SoA |
|---|---:|---:|---:|
| Background activity filter | **1500** | 3652 | 2627 |
| Refractory filter | **312** | 577 | 734 |
| Hot pixel filter | **627** | 681 | 784 |
| Polarity (2-channel counts) | 466 | 526 | **464** |
| Voxel grid (9 bins, 30ms) | **1961** | 3224 | 3354 |
| Time surface (tau=30ms) | **1658** | 1789 | 1811 |

