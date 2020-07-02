# Computer Vision and SET
In this project I attempt to teach a neural net how to recognize a "SET" in the game Set (see https://en.wikipedia.org/wiki/Set_(card_game)).


## Problem Statement

The game of Set involves players finding patterns between cards and finding a "SET" of three cards that have certain attributes in common. Each card has some shapes on it, which can be characterized by four main attributes: the outline of the shapes on the card, the color of the shapes, the number of shapes on the card, and the way the shapes are filled in.

Image classification encompasses a huge subset of computer vision problems today (reading handwriting, labeling x-rays, assessing crop yields, just to name a few). When I look at, say a firetruck for example, how do I know it's a firetruck? Well, I see its distinguishable bright solid-filled red color, its large boxy shape, and the fact that it's one continuous structure. Of course I also see that it has many wheels, and a ladder, etc, but the initial features that tell me what it is are encompassed in the four main attributes of the cards of the game Set. 

This connection inspired my problem statement: Can I train a computer to detect a set, and in doing so draw insights about similar computer vision problems?


## Data Collection and Cleaning

I actually have the game SET at my home and scanned the actual playing cards in to a folder in my computer using my printer/scanner (note: as the game is trademarked by Set Enterprises, the data is not included in the public version of this project).  I turned the scans into a pandas dataframe that had each card's color, number, shape, and fill labeled by myself. I then generated the 85,320 (81 choose 3) sets of three cards, wrote a function using mod 3 arithmetic to decide which were sets and which were not, and saved this as a numpy array to be fed into a neural net.   


## Summary
A convolutional neural net with two convolutional layers, a max pooling layer, two dense layers with relu activation functions and a final dense layer with a sigmoid activation was trained on 2,160 sets of three cards. The best model saw 75% accuracy which was better than the baseline model of randomly guessing, which would see 50% accuracy.  


## Conclusions

Performance was best on color and actually worst on shape. This was a surprise to me, since the cards are white right up until they hit the boundary of a shape on the card, so I thought the neural net would be better at detecting this. However, many of the shapes are similarly oblong, and corners were blurred in the convolutional layers, so this does make sense. 

In the future, a deeper neural net should be explored to improve performance.




