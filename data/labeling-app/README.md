# Podcast Data Labeling App

This is a Shiny for Python data labeling app based on the [suvery form](https://shiny.posit.co/py/templates/survey/) from the Shiny for Python [templates.](https://shiny.posit.co/py/templates/)

You can use this in conjunction with a csv (the one I'm using is gitignored and not part of the repo) to create labeled versions of your data.

I'm using this Shiny app in conjunction with [uv](https://github.com/astral-sh/uv) for Python project/package management. 

If you've cloned the repo run

```uv sync``` 

in the repo root directory to make install the Shiny module for Python into the virtual environment.

THen, once that's happened you can navigate to this directory and run

```shiny run app.py --reload```

The Shiny app should start on port 8000.
