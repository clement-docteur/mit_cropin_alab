# Check by eye: species and clear-cuts at 30 points

A small visual check of two things the species analysis could not settle:

- **SDPT's species labels:** whether its pine and eucalyptus labels are right at our points. In Landsat, SDPT's pine
  and eucalyptus points look identical.
- **The cut detector:** whether the harvest-cycle detector finds the clear-cuts you can see in Google Earth's
  historical high-resolution imagery. Even SDPT "eucalyptus" shows only about one detected cut per 13 years, against
  the expected 6–7 year rotation.

## The 30 points

`../scripts/eye_check_sample.py` draws them, at random within each group with a fixed seed. All are MapBiomas
reference points, which are public.

| Group | Points |
|---|---|
| SDPT pine plantation | 8: 4 caught and 4 missed by the main model (H-all, random forest) |
| SDPT eucalyptus plantation | 8: 4 caught and 4 missed |
| Plantation outside SDPT | 8: 4 caught and 4 missed |
| Natural-forest reference point inside an SDPT plantation that the model calls plantation | all 5 |
| Natural-forest control (outside SDPT, forest every year since 1985) | 1 |

The points are shuffled and named E01–E30, so the order gives nothing away.

## Files

| File | Use |
|---|---|
| `eye_check_points.kml` | Open in Google Earth Pro: a pin and a yellow ~30 m square (the Landsat pixel, centred on the point) at each point |
| `eye_check_points.csv` | The same points with Google Earth and Google Maps links |
| `eye_check_key.csv` | What our data says at each point: SDPT species, MapBiomas labels, the model's call, detected cut years. Open it only after recording |
| `eye_check_data.json` | Everything above plus each point's yearly NBR, for the recording page |

## How to record

Answers go in the recording page, which shows our data for a point only after it is saved. For each point:

1. In Google Earth Pro, turn on historical imagery (the clock icon) and step through every date.
2. Record what is in the yellow square in the most recent image.
3. Record the year of the oldest sharp image, and what was in the square then.
4. Mark each year a clear-cut or thinning first shows.
5. Rate how sure you are of the species and of the cut years.

Answers are stored with the page, where Claude can read them back to tally:

- species agreement with SDPT;
- cuts seen vs cuts detected, from the oldest sharp image to 2020;
- what the plantations outside SDPT, and the "forest" points inside SDPT, actually are.
