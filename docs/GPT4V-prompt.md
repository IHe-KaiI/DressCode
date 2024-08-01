The following prompt is used in GPT-4V for data captioning.

```
I will provide 4 rendered results of several virtual garments with a human model, each image containing two views of a garment, the left one is the front view of this garment and the right one is the back view.

For each virtual garment, you should generate TWO strings.

In the first string, describe the garment type (If THE SUBJECT HAS A NAME, INCLUDE ITS NAME FIRST!);

  Example phrases for the first string: "hood", "T-shirt", "jacket", "tuxedo", etc.


In the second string, describe the overall global geometric features of the garment (DO NOT INCLUDE ANY INFO ABOUT THE HUMAN MODEL AND THE COLOR INFO OF THE GARMENT) using several different short phrases split by ',' with the following tips: 

  Example rules:
  	Describe the length of the sleeves: long, normal, short, sleeveless, etc.
  	Describe if it has a hood: with a hood, etc.
  	Describe the length of the dress: long, normal, short, etc.
  	Describe the width of the garment: wide, normal, narrow, etc.
  	Describe the length of the legs of trousers: long, normal, short, etc.

  Please follow the example rules above (not limited to these examples) to describe the geometric features of the garment.

  Example phrases for the second string: "long sleeves", "wide garment", "with a hood", "deep collar", "sleeveless"...

Please strictly avoid mentioning color, texture, and material.

Return 4x2 strings in ONLY a nested JSON list.
```

Along with the prompt, we will upload 4 rendered images from the dataset to GPT-4V, each image is concatenated by two views of the garment.