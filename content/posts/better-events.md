+++
date = '2026-09-24T14:24:35+01:00'
title = 'Event-based data: a better way to store events'
draft = true
+++

TL;DR: How to save 70% memory by choosing another intermediate representation for event-based data.

## Intro 

Event cameras generate an asyncrhonous stream of events, basically what you get out of the camera is a list of events, where one event consists of x,y coordinates, a timestamp and a polarity (boolean ON or OFF).

In Prophesee's OpenEB library, an event is stored as:

```cpp
struct Event {
    uint16_t x;
    uint16_t y;
    int16_t p;
    uint64_t t; // timestamp in microsecond
```

And the API gives us in a callback some `std::vector<Event>`.


Already we can see the layout is not great:

```
offset  size
0       2     uint16_t x
2       2     uint16_t y
4       2     int16_t  p
6       2     padding
8       8     uint64_t t
------------------------
total  16 bytes
```

One event is 16 bytes! That's 128 bits or 2 uint64_t. An event camera can generate millions of events per seconds, so the bandwitdh can become the bottleneck of every application.


## How can it be improved


### Pack stuff

Packing will save us a lot of memory, but it will come at the cost of unpacking. But let's give it a go:

Let's say our format will support sensors up to 2048x2048, that's a limitation but it also mean we can use only 11 bits for each coordinate, and because we need one bit for the polarity, the proposed layout is:

```
bit: 31                      23 22 21          11 10               0
     +-------------------------+--+---------------+----------------+
     |       unused (9)        |P |     Y (11)    |    X (11)      |
     +-------------------------+--+---------------+----------------+
```

I can already hear you scream: "But it's terrible, you will have to unpack these everytime!!". The key idea: You don't! You don't have to unpack it, you can treat this as a single number that can index into an array.

Let's call this value the EventIndex, or EventID. It doesn't depend on time, it identifies an event coordinates and polarity. Let's say you want to build a time surface (which is just a map of the most recent activity).


```
const W: usize = 1280;           // sensor width
const H: usize = 720;            // sensor height
const STRIDE: usize = 2048;      // the id's own row stride
const TAU: f32 = 50_000.0;       // decay constant, microseconds

// Surface of active events (SAE), stores the last timestamp for each events.
// Here we ignore polarity, but we could have a 2-channel SAE, one for each polarity
let mut sae: Vec<u32> = vec![0; H * STRIDE];

// Keep the latests timestamp
let mut latest_ts: u32 = 0;

// The output: A tensor in sensor coordinates, ready for the network.
let mut surface: Vec<f32> = vec![0.0; H * W];


// ── Thread A ── camera ──────────────────────────────────────────────
// Write the SAE. Never allocates, never blocks, never reads.
loop {
    for (id, ts) in camera.next_events() {
        // The id packs polarity, y and x. Masking off the polarity bit
        // leaves y * 2048 + x -- already a row-major pixel index, so the
        // event IS the address. No unpacking, no y * width + x.
        sae[(id & PIXEL_MASK) as usize] = ts;
        latest_ts = ts;
    }
}


// ── Thread B ── inference ───────────────────────────────────────────
// Every 20 ms, decay the SAE into a tight [H][W] tensor.
loop {
    sleep(20.ms);
    let t_now = latest_ts;

    for y in 0..H {
        let src = &sae[y * STRIDE ..][..W];        // SAE row: 2048 wide, first W used
        let dst = &mut surface[y * W ..][..W];     // tensor row: W wide, sensor coords

        // Contiguous -> contiguous, so this vectorises. Changing stride
        // costs nothing: it lives in the slice bounds, not the inner loop.
        for x in 0..W {
            let age = t_now.wrapping_sub(src[x]) as f32;
            dst[x] = (1.0 - age / TAU).max(0.0);   // linear decay
        }
    }

    run_network(&surface);
}
```


So basically by storing both X and Y, row-major, on 11 bits we are using a stride of 2^11, i.e 2048. This is great but we can actually adapt this stride according to our needs. For instance if we want to index directly into an array we can use y*width+x. In this case our layout becomes:

```
 Adapted stride (=width)  id & PIXEL_MASK  ==  y * width + x

  31            23 22 21                             0
 ┌────────────────┬──┬────────────────────────────────┐
 │      free      │p │          y * width + x         │
 └────────────────┴──┴────────────────────────────────┘
                      └───────────────┬───────────────┘
                                      └─ 22 bits, no field boundary
```

If we want to do branchless neigborhood check, for instance if we want to do a background activity filter by looking at the 8-neigbor of events, we can use STRIDE=width+2, to add two more columns, and add two more rows in our arrays, this way an algorithm that checks the neigbors doesn't have to do `if (x>WIDTH-1) etc...`, we can efficiently do: 

```rust
  const S: i32 = (WIDTH + 2); // S is the Stride
  let pixel_index = (y + 1) * S + (x + 1);
  // check whether any of the 8 neighbours fired less than DT ago
  for &o in &[-1, 1, -S, S, -S-1, -S+1, S-1, S+1] {
      let n = sae[(pixel_index + o) as usize]; // Get neighbor values, no if needed
      hit |= t.wrapping_sub(n) < DT; // Build a mask
  }
```

### The EVT3 wire format bonus! 

When the data comes out of the camera, it's actually encoded in a more compact representation, usually EVT21 or EVT3. The first step is to decode this data.

EVT32 uses 16 bits words where X and Y are already encoded on 11 bits.



### Wrapping up

```
//! The decoded event representation.
//!
//!   bit 22 = polarity | bits 21:0 = pixel = y * STRIDE + x
//!
//! STRIDE is a const generic, defaulting to 2048. That default is not a design
//! choice: EVT3 already carries x and y as separate 11-bit fields, so a stride
//! of 2^11 means packing an event is one OR and the decoder never needs to know
//! the sensor geometry. Hence `NativeId`.
//!
//! Making STRIDE generic rather than a runtime field is what buys the freedom
//! to pick another one. From the generated assembly:
//!
//!   STRIDE = 2048     x() -> `and $2047`      y() -> `shr $11; and $2047`
//!   STRIDE = 1282     x() -> multiply-shift   y() -> multiply-shift
//!   runtime stride    x() -> `divl`           y() -> `divl`
//!
```

It's just a wrapper around a `u32` value.

```
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
#[repr(transparent)]
pub struct EventId<const STRIDE: u32 = 2048>(u32); // Use 2048 as default stride
```

We can make an alias if we want: 

```
pub type NativeId = EventId<2048>;
```

Here's an example implementation, I ommited `#[inline(always)]` and tried to keep it simple.

```
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

    /// Row-major pixel index, polarity stripped.
    pub const fn index(self) -> u32 {
        self.0 & Self::PIXEL_MASK
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

We can now define a PixelMap, a per-pixel array that an `EventId` indexes directly.

```
pub struct PixelMap<T, const STRIDE: u32 = 2048> {
    data: Vec<T>,
    width: u32,
    height: u32,
}
```

I will not go into the implementation details of such map but it can be used to create surfaces of active events, accumulators, histograms etc... 



## What about timestamps ? 


The orginal OpenEB `Event` stores timestamp as a u64 microseconds count. 64 bits is 
18446744073709551616 microseconds, that's around 584 942 years !! It's extremely wastful.

I suggest to use only 32 bits for the timestamp, which gives us around 71 minutes. That's not much, but we can always store an absolute time offset alongside our recordings, and split them by chucks of 71 minutes, and inside an application we can detect the warping and keep track of a seperate counter.


With our u32 x,y,p and u32 timestamp, we are now using 8 bytes! We divided by two the memory used by OpenEB, but we can do better!

## Come on! Don't store all timestamps


Multiple events triggers at the same timestamps, why bother storing the same timestamp everywhere, what we could do instead is *store only when the timestamp changes*.

The idea is simply to store a `TimeMark` that is:

```
struct TimeMark {
    ts_us: u32,       // timestamp, µs relative to the buffer origin
    event_idx: u32,   // index of the first event with this timestamp
}
```

And store event buffers as event ids (x,y,p) and time marks. 
```
struct EventBuf {
    ids:   Vec<EventId>,    // 4 Bytes per event
    marks: Vec<TimeMark>,   // 8 Bytes per *distinct timestamp*
    t_origin: u64,          // absolute start, stored once
}
```

For this to work, we need to have long *segments* of events with the same timestamp. This is very data-dependant, but on my tests I have seen:

```
length    segments  %segs  %events   mark cost B/event
1          1561587  14.9%    1.4%    8.00
2          1044718   9.9%    1.9%    4.00
3-4        1113288  10.6%    3.4%    2.36
5-8        2055961  19.6%   12.5%    1.18
9-16       2295553  21.8%   25.5%    0.65
17-32      2035018  19.4%   40.2%    0.36
33-64       402033   3.8%   15.1%    0.19
65-128        1528   0.0%    0.1%    0.12
```

Which gives: 

| Layout | Bytes/event |
|---|---:|
| vector of OpenEB `Event` | 16.00 |
| event_id + `u64` per event (OpenEB-ish) | 12.00 |
| event_id + `u32` per event | 8.00 |
| **id + 8 B marks** | 4.75 |


Which mean we can go from 16 Bytes per event to 4.75 Bytes per event! **The new representation is a 70% memory reduction versus orignal `vector<Event>` in openeb!**. But wait to see what comes next ! 


If we compare it to the baseline EVT3 wire format, we can see:

| Representation | Size | Bytes/event | vs raw EVT3 |
|---|---|---|---|
| `vector<EventCD>` | 1,783.8 MB | 16.00 | 4.86x |
| `EventId` + `TimeMarks` | 530.0 MB | 4.75 | 1.44x |
| Raw EVT3 | 367.3 MB | 3.29 | 1.00x |

We use only 1.44x while beeing fully decoded and so much more convinent for algorithms

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


// A buffer of event is a list of ids, time marks, and a time offset. 
struct EventBuf {
    ids:   Vec<EventId>,    // 4 Bytes per event
    marks: Vec<TimeMark>,   // 8 Bytes per *distinct timestamp*
    t_origin: u64,          // absolute start, stored once
}
```

## What about the consumers ? 

I hear you cry: "But man ! Now I have to unpack my EventIDs to extract x,y,p and I have to deal with weird TimeMarks".

I already tried to show that you don't necessarly to unpack the EventID, and the goal of this single `u32` is to use it to index directly into 1D arrays.

For the timestamp, the `TimeMark` design allows to iterate over **segments**. And this is extremely useful and **More efficient** for certain processing. Here are some examples, with benchmarks:



## Can we use it as a storge format ?

Serde/Zstd


