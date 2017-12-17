# UFC MMA Predictor

The UFC MMA Predictor https://ufcmmapredictor.herokuapp.com/ is a web app to predict fights exclusively in the Ultimate Fighting Championship (UFC). Please use at your own discretion as there is no guarentee of profit. **The machine learning model behind this web app currently gives a 70.4% accuracy**. This is also considered a professional project of mine. Please read the [Jupyter Notebook documentation](https://github.com/jasonchanhku/UFC-MMA-Predictor/blob/master/UFC%20MMA%20Predictor%20Workflow.ipynb) for more detailed information on the data preparation, modelling, and visualization.

Here you will find the following documents:

* [Jupyter Notebook](https://github.com/jasonchanhku/UFC-MMA-Predictor/blob/master/UFC%20MMA%20Predictor%20Workflow.ipynb) of data science pipeline for the project
* Python scripts to the web app
* R web scraping scripts 
* Demo of the app

# Getting Started

Here are a few information you need to have before using the web app:

* Fight card information (who is fighting who)
* Fighter information (weightclass and name)
* Betting odds (preferred decimal odds), knowing who is the favourite and underdog fighter beforehand

## Things to note

Here are some of the trends that have been spotted in recent MMA years to note:

* Picking favourite only gives 60% chance of winning in long term
* Fights go on longer in general 
* Southpaws win 53% of their fights, as opposed to 48% for orthodox fighters 
* Fighters who switch stances have an even higher success rate than both orthodox and southpaw fighters, with a combined win percentage of 57%

# Demo

![](https://github.com/jasonchanhku/UFC-MMA-Predictor/blob/master/Pictures/demo.png)

Let's take the main card of **UFC 217** to see how well the app performs. Note that UFC 217 fight data is neither in the training or test set. The fights that happened were (winners in **bold**):

* **GSP** (@2.10) vs Michael Bisping (@2.00)
* Cody Garbrandt (@1.57) vs **TJ Dillashaw** (@2.75)
* Joanna (@1.20) vs **Rose Namajunas** (@7.00)
* **Stephen Thompson** (@1.57) vs Jorge Masvidal (@2.57)

The web app predicted all the above fights correct except for **Joanna vs Rose Namajunas** where Joanna had most fight stats dominant over Rose and was the big favourite over Rose. However, Rose caught Joanna with a fight ending hook during the fight and it was indeed 'Goodnight Irene'.

# Citation

I would appreciate it if you cite my [Github repo](https://github.com/jasonchanhku/UFC-MMA-Predictor) if you ever use any information or derived any inspiration from here.
