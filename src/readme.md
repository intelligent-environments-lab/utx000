# Hagen-Flavored Programming
This document functions as documentation for the syntax used in Python files. Knowing the conventions used in the coding will help when trying to decide how the inputs to different functions should be or even what the names of the functions are. 

While there are unique aspects to Hagen-Flavored Programming, the style is taken mostly from the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style for Python programming. 

## Indentation and Spacing
All ```.py``` files are indented with spaces - 4 spaces per level. The Jupyter notebook files (.ipynb) are, by default, indented with tabs.

Blank lines are used sparingly and consistent with the PEP 8 style.

## Variable Names
The basic naming convention is outline for various aspects of Python programming in the table below.

| Aspect | Naming Style | Example | Exceptions |
| --- | --- | --- | --- |
| Variable | mixed case | ```beiweSleepDF``` | As the example shows, all letters in acronyms are capitalized |
| Function | lowercase with underscores | ```get_fitbit_sleep_metrics()``` | None |

## Figure Color Palette
The base colors used for visualizations are enumerated below and are available from [Matplotlib](https://matplotlib.org/3.1.0/gallery/color/named_colors.html). 

| Primary Use | Secondary Use |
| --- | --- |
| black | grey |
| firebrick | darkorange |
| gold | khaki |
| seagreen | lawngreen |
| cornflowerblue | dodgerblue |
| darkorchid | plum |

The following table highlights the nuanced uses of the base colors described above.

| Plot Type | Color | Notes |
| --- | --- | --- |
| Single Var Plot | cornflowerblue or black | |
| Comparison | firebrick and seagreen | firebrick for negative components and seagreen for positive |
