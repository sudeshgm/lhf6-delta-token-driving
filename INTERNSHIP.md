# Internship project — Summarising a drive one token per frame

**Duration** 3 months · **Hardware** one 16 GB GPU · **Data** open-source datasets only

---

## 1. What this project is about

Video is expensive to represent because almost every method turns each frame into hundreds of
numbers, then has to deal with thousands of frames. A recent idea sidesteps this neatly: instead
of describing each frame, describe only **what changed** since the previous one, and compress that
change into a single vector per frame. A whole video becomes a short sequence rather than a large
grid, roughly a thousandfold reduction, and it still performs well on forecasting tasks — because
most of a scene does not change and re-describing it every frame is wasted effort.

That idea was demonstrated on general video over short horizons. You will find out whether it
works as a compact summary of **driving** footage — good enough to group similar situations
together and search for them.

There is a specific reason this is not obvious, and it is the heart of the project. In a driving
video, nearly all the frame-to-frame change is caused by our own vehicle moving forward past
stationary things: buildings, poles, road markings. A naive change-based summary may therefore
encode little more than how fast we were going. Working out whether that is happening, and
removing it if so, is the real work here.

## 2. Why it matters

A compact, searchable summary of recorded video is useful for grouping similar situations,
retrieving comparable recordings and cheaply indexing a large archive. Doing it with one vector
per frame, on frozen features, is cheap enough to run over a whole archive.

## 3. What you should already know

**Needed**
- PyTorch; extracting features from a frozen vision model and training a small network on top.
- Basic multi-view geometry: what a camera pose is, what parallax means, why a scene point's
  apparent motion depends on its distance. You will need this for the ego-motion part.

**Useful, not required**
- Clustering and retrieval evaluation.
- Any driving dataset.

**Not needed**
- Generative modelling experience — this project uses the representation, not the generator.

## 4. What you will build

**Part of the project**
- A change-based tokeniser on top of a frozen vision backbone, producing one vector per frame.
- Three variants: raw change, change after compensating for our own vehicle's motion, and change
  restricted to independently moving regions.
- Evaluation by retrieval and clustering against situation labels.
- A test of whether quality holds up on long recordings as well as short ones.
- A storage-versus-accuracy comparison against simpler alternatives.

**Not part of the project**
- Video generation or future prediction, which is what the original method was for.
- Pixel-level reconstruction quality.
- Training the vision backbone.

## 5. Getting started — week 0

**Reading**
1. *A Frame is Worth One Token: Efficient Generative World Modeling* — the representation you are
   adapting.
2. A short refresher on epipolar geometry — the fundamental matrix and what the Sampson distance
   measures. You will use this rather than a simpler planar model, for reasons explained below.
3. Documentation for the driving datasets you will use.

**Setup:** a frozen self-supervised vision backbone, a driving dataset with situation labels, and
a longer-recording dataset for the length experiment.

**First hands-on task (day 4–5):** compute change vectors for 200 clips and plot their magnitude
against our own vehicle's speed. If the two track each other closely, you have identified the
central problem on day five, which is exactly where you want to find it.

## 6. Milestones

| | Week | Target | Done when |
|---|---|---|---|
| **M1** | 3 | Tokeniser working. The relationship between the representation and our own speed measured. | You can state, with a number, how much of the representation is explained by our own motion. |
| **M2** | 6 | All three variants built and compared on situation retrieval. | Compensated and uncompensated versions compared directly, on the same task. |
| **M3** | 9 | Retrieval and clustering evaluated against several simpler alternatives. | Every comparison method has a number, including the deliberately trivial ones. |
| **M4** | 12 | Long-recording behaviour and the storage comparison. Report and talk done. | Short and long recordings indexed together, with any bias between them measured. |

## 7. Week-by-week

| Week | Focus |
|---|---|
| 0 | Reading; first speed-versus-change plot. |
| 1 | Feature extraction and the change tokeniser. |
| 2 | Measure directly how well our own speed and turn rate can be predicted from the change vectors. A high number means the representation is essentially a speedometer. |
| 3 | Build a control set of clips where our vehicle is stationary at a junction with traffic crossing. Change vectors should work well here and poorly on open-road cruising; if that pattern appears, the diagnosis is confirmed cheaply. **M1** |
| 4–5 | Compensate for our own motion. Use epipolar geometry rather than a single planar transform: driving forward past buildings and poles produces parallax that a planar model cannot explain, and fitting one anyway will mark every building as a moving object. |
| 6 | Compare all three variants on retrieval. **M2** |
| 7–8 | Full retrieval and clustering evaluation, with comparison methods: average of frame features, single middle frame only, simple hand-crafted motion statistics, and a frozen image-text baseline. |
| 9 | Control experiments — shuffle the change sequence (does order matter?), and a one-dimensional baseline using nothing but our own speed over time. If that alone does well, say so; it is important information. **M3** |
| 10 | Index short and long recordings together. Check whether long recordings are systematically favoured or ignored — a known weakness of representations that depend on input length. |
| 11 | Storage against accuracy, versus the alternatives. |
| 12 | Report, presentation, handover. **M4** |

## 8. What you will deliver

1. **Tokeniser and evaluation code.**
2. **The ego-motion analysis** — how much of the representation is our own movement, and how well
   compensation removes it. This is likely the most useful single result.
3. **Retrieval and clustering results** against all comparison methods.
4. **Report, 10–15 pages**, with a recommendation on whether this representation is worth using
   for driving archives.
5. **Presentation, about 20 minutes**, to the team.

## 9. Working rhythm

- Weekly one-to-one, 45 minutes.
- Short written status every Friday.
- Demo to the wider team at M2 and M4.
- If you are stuck for more than two days, say so.

## 10. If something does not work

| Situation | Plan |
|---|---|
| The representation is essentially a speedometer, and compensation does not fix it. | Report it clearly. This is directly useful to anyone else considering this representation for driving footage, and it is a legitimate outcome rather than a failure. |
| The speed-only baseline does as well as the full method. | The retrieval task is too easy. Make it harder by using queries where our own behaviour is held constant and only the surrounding situation varies. |
| Length invariance does not hold. | Drop that part and focus on the short-recording results, which stand on their own. |
| Extracting features for long recordings is too slow. | Reduce the frame rate for that experiment and record the choice. The question is about the representation, not the sampling. |

## 11. What you will take away

Experience taking a representation from one domain and finding out honestly whether it transfers
to another — including the diagnostic work of identifying what it is *really* encoding. Practical
multi-view geometry. And a good example of why the first plot you make (change against speed)
often matters more than the last one.
