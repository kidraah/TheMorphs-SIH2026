# Importance and Necessity of CartoDEM

## 1. Water Obeys Gravity
In cloudburst prediction, atmospheric data (IMDAA/INSAT) only tells us that 150mm of water will fall from the sky. It does *not* tell us what happens when it hits the ground. 
Without CartoDEM, we cannot predict flash floods. Water flows downhill. A DEM is the absolute mathematical rulebook for gravity in a computer simulation.

## 2. The Orographic Lift Catalyst
Why do cloudbursts happen mostly in the Himalayas? **Orographic Lift.**
When humid monsoon winds hit a steep mountain, the wind cannot go through the rock, so it is forced violently upwards. This rapid lifting cools the air instantly, squeezing out all the water in a massive cloudburst.
Without CartoDEM, our AI does not know where the mountains are, how steep they are, or which direction they face. By feeding CartoDEM into the AI, the neural network "sees" the mountain and learns that wind + steep slope = disaster.
