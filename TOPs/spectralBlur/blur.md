# minding the blur
### Frequency-Domain Convolution

Inspired by [this video](https://www.youtube.com/watch?v=ml-5OGZC7vE) — go watch it to hear from an actual expert :)

---

## What is Blur?

At its core, a blur removes fine detail while preserving the larger shapes and forms of an image. A useful way to think about it: **blur is a Low Pass Filter** —  it strips out high-frequency detail (sharp edges, fine detail) while letting through the low-frequency content (broad shapes, gradual colour transitions). Let's keep the idea of a Low Pass Filter in the back of our mind and we will get back to it.

---

## How Does Blur Work?

The classic approach uses a **kernel**: each pixel looks at its surrounding neighbours and averages their values. You can get fancy with how you weight those values or use different kernel shapes — but the underlying principle is always the same: sample a neighbourhood and average it out.

This works well, but a large blur requires *a lot* of samples, which gets expensive fast. You can downsample the image at different levels to help, but that introduces its own creative limitations — and changing the kernel shape is a whole separate headache.

---

### But hmmm... Low-Pass Filter??

Since blur is a Low Pass Filter, what if we could split the image into a **frequency spectrum** and simply mask out the high frequencies directly?

This would mean:
- A massive blur costs exactly the same as a small one
- You get full artistic control over the frequency spectrum
- You can create different shapes — streaks, hexagons, custom and kernels trivially

And then just convert that frequency-representation back to a regular image-representation?

Would that be smarter? maybe - the issue is that Converting an image to spectrum-space and back is really, really, really difficult and pretty expensive. 

But as most things in TouchDesigner - the derivative devs have already done most of the heavy lifting for us. And we can leverage the **`Spectrum TOP`**

This allows convert a channel, red, green, or blue from image space to frequency space with a discrete fourier transform. and then we can convert it back to image-space with an inverse fourier transform.

---

## The Signal Chain

```
Spectrum TOP (DFT)  →  Convolution Pass  →  Spectrum TOP (Inverse DFT)
```

1. **`Spectrum TOP`** — converts a channel (R, G, or B) from image space to frequency space
2. **Convolution pass** — multiply the magnitude (red channel) by a 2D shape (e.g. a circle)
3. **`Spectrum TOP`** — inverse DFT converts it back to a viewable image

### Reading the Spectrum

The spectrum is laid out in **polar coordinates**:

| Channel | Contains | Description |
|---|---|---|
| Red | Magnitude | Amplitude of each frequency |
| Green | Phase | Geometric information — we will not touch this today|

- **Middle** → low-frequency content (broad, slow-changing forms)
- **Edges** → high-frequency content (sharp transitions, fine texture, noise)

To blur, simply draw a circular mask and multiply it with the spectrum — keeping the centre, discarding the edges — then transform back. That's it.

The main limitation - apart from performance - here is that when working in frequency space, **the effect applies to the whole texture uniformly**. Variable blur (sharp in one corner, blurry in another) isn't possible with this approach. But we can deal with that in post.

---

## Using the Component

[Download the component here](https://github.com/supermarket-sallad/TD-components-shared/tree/main/TOPs/spectralBlur)

### Parameters

It has some pre-made convolution kernels under the BLUR page - that you can view with the "viewConvolutionKernel" - output
You can also use your own kernel by linking a TOP to the "Custom Convolution Kernel" 

Things to be aware of: This is an expensive effect. in order to keep it a bit smoother - I downsample the image before coonverting it to a frequency. You can change by how much in the "Spectrum Pass Resolution" parameter.

If you want this to be a Bloom rather than a blur - you can use the PREPROCESS page to composite the blur with the input. here - you can also set the pre blacklevel and brigtness. you can also use the desaturate bright points to create more of a glow - since really bright light tend to shift toward light.

if you want a (kind of) luma blur - you can link a TOP to the "Attenuate Map" parameter.

>  **Performance note:** This is an expensive effect. Lowering the Spectrum Pass Resolution will help keep things smooth.

---

##  Known Issue

The `Spectrum TOP` has been acting strangely lately — flickering and exploding values, especially when middle-mouse-clicking on operators. You might need to turn off and on the component. I've filed a bugreport about it. **Use this component with caution for now.** and see exxample file for some ways to mitigate it.

---

Have fun exploring — the frequency domain is a genuinely interesting space to play in.