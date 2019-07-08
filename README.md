# A Comparative Study of Statistical-based Change Detection Methods for Multidimensional and Multitemporal SAR Images

The work here correspond to a comparative study of various methods based on parametric statisticial analysis for change detection in multidimensional and multitemporal SAR images.
The aim is to allow reproducibility of the study published in the paper presented at the World Congress on Condition monitoring 2019 in Singapore.

## Files' organisation

This folder is organised as follows:
- **change_detection_functions.py**: Contain the codes for every detector compared.
- **Comparative Study Change Detection SAR.ipynb**: a Jupyter notebook for the study on real dataset.
- **Constant False Alarm Rate Testing.ipynb**: a Jupyter notebook for testing CFAR property.
- **Time Consumption.ipynb**: A Jupyter notebook for comparing time consumptio of methodologies.

## Requirements for Python
	The code provided was developped and tested using Python 3.7. The following packages must be installed 
	in order to run everything smoothly:
	- Scipy/numpy
	- matplotlib/plotly
	- tqdm

The code use parallel processing which can be disabled by putting the boolean enable_multi to False in each file.
The figures can be plotted using Latex formatting by putting the boolean latex_in_figures to True in each file (must have a latex distribution installed).

Dataset available at https://uavsar.jpl.nasa.gov/. For ground truths, please contact the authors.

## Credits
**Author:** Ammar Mian, Ph.d student at SONDRA, CentraleSupélec
 - **E-mail:** ammar.mian@centralesupelec.fr
 - **Web:** https://ammarmian.github.io/
 
 Acknowledgements to:
 - [**Frederic Pascal**](http://fredericpascal.blogspot.com/), Laboratoire des Signaux et Systèmes, CentraleSupélec, 3 rue Joliot Curie, 91192 Gif-sur-Yvette, France

## Copyright
 
 Copyright 2019 @CentraleSupelec

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.