# Emulates a 2-Drum Riso Printer

On a Riso printer, you can only print with 2 inks
(you _could_ do more, but usually there are just 2 drums in there).

So the basic concept is:
what if we only had two colors (inks) to work with?

You can set up these two colors under the **COLORS** tab.
We also need to know how to weight the input image — this is done with the CMYK weights.

You can toggle **Get Automatic CMYK Weights**.
This will try to match the input image to your color selection.

In order to emulate ink, this works in **CMYK color space**.

---

## COLORS TAB

- **RGB Color 1** – ink color 1
- **RGB Color 2** – ink color 2

Here you can set the two ink colors we will use.

---

### Get Automatic CMYK Weights

Toggle this to automatically get CMYK weights.
If enabled, it updates automatically when you change colors.

---

### CMYK Weights for Color\*

Here you set how much the CMYK components contribute to each color.

---

### Output Printable Passes

If you're planning on actually Riso-printing these, you will need two monochrome rasters:

- **Color 1** → red channel
- **Color 2** → green channel

You can use a reorder TOP to separate them.

---

## TOTONE TAB

Here you can set the halftone type.

### Halftone Type

- **Halftone** – creates a halftone pattern similar to printers
- **Bayer Dither** – emulates old-school digital print and displays; may create intense moiré patterns if the output resolution differs from the display
- **Threshold** – flat poster-style output
- **None** – no halftoning, only color weighting

---

### Halftone Parameters

_(only available if Halftone Type = Halftone)_

- **Grid Scale** – number of dots relative to vertical resolution
- **Random Scatter** – randomly offsets points
- **Rotate Layer 1 / 2** – rotates each pattern grid independently
- **Halftone Bleed** – how much points bleed into neighbors
- **Halftone Threshold** – lifts the minimum dot size
- **Shape** – select from different shapes
- **Style** – soft or hard halftone
- **Sample Exact Cell** – disable for line shapes; affects how the input is sampled

---

### Threshold Parameters

_(only available if Halftone Type = Threshold)_

- **Threshold 1 / 2** – cutoff values (independent per layer)
- **Smoothing 1 / 2** – smoothing amount
- **Grit** – adds noise

---

## TRANSFORM TAB

- Offset, rotate, and scale each layer independently

---

## PRE-PROCESS TAB

- Set brightness adjustments **before** any other processing

---
