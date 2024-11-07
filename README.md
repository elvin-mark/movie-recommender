# Movie Recomender

Simple movie recommender that leverage a simple light-weight text embedding deep learning model to search for similar movies from a clustered movie list.

# How to run

Run the following command to start serving

```sh
streamlit run recommender.py
```

Then open this URL on your browser [http://localhost:8501/](http://localhost:8501/) (by default it will automatically open this url in your default browser)

# Reference

- [Movies Dataset](https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset?select=movies_metadata.csv)
- [Feature Extraction Model](https://huggingface.co/Supabase/gte-small)
