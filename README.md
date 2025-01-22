# Analyses of the Dunc'd On Basketball Podcast

## A demo of the production app predicting the best episodes

https://github.com/user-attachments/assets/fe6e4955-83fd-4d00-8aac-c81388fe9290

The Dunc'd On Podcast only has one problem: they put out too much good content.

So, I build an app that would predict which recent episodes I'm most likely to enjoy.

This [production app](https://duncd-on-predictor-44887993798.us-central1.run.app/) hits the Dunc'd On RSS feed for episodes over the past week, performs feature engineering, and runs a model that achieved an F1 score of 0.80 for predicting the best episodes (as labeled by me). You can try the app yourself [here](https://duncd-on-predictor-44887993798.us-central1.run.app/).

## A demo of exploring episode description embeddings

https://github.com/user-attachments/assets/e11f5792-1330-47ee-aec9-75df8108f8c5

## Progress so far

Hello! So far I've created a couple of data visualizations showing:
- Nate and Danny release episodes at the wildest times of day
- A way to determine how similar or different episodes are based on embeddings from [ModernBERT](https://huggingface.co/blog/modernbert). Now with episode types included and an interactive element (see video example above)
- A breakdown of whether I thought the episode was a banger by the type of episode (Those features were created using the data labeling app I talk about below)

I've also created a labeling app using Shiny for Python where I can categorize each episode and designate whether or not I think the episode was a banger. I'm labeling episodes using this tool in service of creating models that can categorize the kind of episode and predict whether I'll think it's a banger or not.

The current culmination of this repo is the production app described in the first README section!

I may do more development on this project in the future, so if there's something you'd be interested in feel free to file an issue or post at me on [Bluesky](https://bsky.app/profile/mcmullarkey.bsky.social)