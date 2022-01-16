## title: "Summer Project"
## author: "Ping Zhang"
## date: "1/16/2022"


## Part 1: Exploratory Analysis

install.packages('ggvis')
install.packages('tidyverse')
install.packages('ggplot2')
install.packages('dplyr')
install.packages('knitr')
install.packages('stringr')
install.packages('DT')
install.packages('data.table')
install.packages('lubridate')
library(ggvis)
library(tidyverse)
library(ggplot2)
library(lubridate)
library(gbm)
library(tree)


## Read the dataset and get high level understanding

Auto <- read.csv("Dataset.csv",header=T, na.strings='NA')
head(Auto) #six first observations
tail(Auto) #six last observations. Now we cans see there is 'NA'.

class(Auto) #see the class
str(Auto) #see the structure of data frame


## Convert character variables to factors

colnames(Auto)[which(names(Auto) == "ï..Brand")] <- "Brand" # Change column name to keep consistent
Auto$Brand <- as.factor(Auto$Brand)
Auto$Body <- as.factor(Auto$Body)
Auto$Engine.Type <- as.factor(Auto$Engine.Type)
Auto$Registration <- as.factor(Auto$Registration)
Auto$Model <- as.factor(Auto$Model) #312 levels

## Drop "model" column

# Model column is categorical variable and having 312 unique values, 
# which implies, after converting it to dummy, 
# it will add 312 new columns to the dataframe, 
# so we will drop this column.
Auto <- Auto[,-9]


## convert 'Year' into 'Age'
## this dataset is collected in 2019, so the age is the interval between 2019 and column 'Year'

# For column 'Year', we also want to convert it into 'Age'
Auto$Year = years(Auto$Year) # use lubridate package to change data type of column 'Year'
a <- years(2019) # calculate interval between 2019 and column 'Year'
Auto$Age <- year(a - Auto$Year) # extract interval between 2019 and column 'Year'
Auto <- Auto[,-8] # drop column 'Year'


## Check the summary result
summary(Auto)

## Check missing values
colSums(is.na(Auto))

## Drop observations with missing value
Auto_clean <- Auto[complete.cases(Auto),] 
na.omit(Auto_clean)
Auto <- Auto_clean


## Viewing trends in price, mileage, enginev, age
par(mfrow=c(2,2))
plot(density(Auto$Price), main='Price Density Spread')
plot(density(Auto$Mileage), main='Mileage Density Spread')
plot(density(Auto$EngineV), main='EngineV Density Spread')
plot(density(Auto$Age), main='Age Density Spread')


## From above, we can say that price and enginev are not normally distributed, so we need to remove some outliers from data.
car_data1 <- Auto[Auto$Price < 161000, ]
car_data2 <- car_data1[car_data1$EngineV < 8, ]
car_data_modified <- na.omit(car_data2)

str(car_data_modified)
summary(car_data_modified)

## Use the pairs() function to produce a scatterplot matrix
pairs(car_data_modified)


par(mfrow=c(2,2))

plot(car_data_modified$Mileage, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Mileage),
     ylim = range(car_data_modified$Price)
)

plot(car_data_modified$EngineV, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$EngineV),
     ylim = range(car_data_modified$Price)
)

plot(car_data_modified$Age, car_data_modified$Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Age),
     ylim = range(car_data_modified$Price)
)


## From above plot, we can say that relationship is not linear in any of the case, so for now we cannot apply linear regression, first of all, we have to do some changes in the dataset.

library(tidyverse)
library(caret)
library(rpart)
library(ipred)
library(parallel)
library(e1071)
library(randomForest)

my_ggtheme <- function() {
  theme_minimal(base_family = "Asap Condensed") +
    theme(
      panel.grid.minor = element_blank(),
      plot.title = element_text(vjust = -1)
    )
}

car_data_modified %>% 
  ggplot(aes(x = Price)) + 
  geom_histogram(fill = "dark orange") + 
  labs(
    title = "Histogram: Price",
    x = "Price",
    y = "Frequency"
  ) + my_ggtheme()


car_data_modified$log_Price <- log(car_data_modified$Price)

car_data_modified %>% 
  ggplot(aes(x = log_Price)) + 
  geom_histogram(fill = "dark orange") + 
  labs(
    title = "Histogram: Log Price",
    x = "Log Price",
    y = "Frequency"
  ) + my_ggtheme()


## For normality and homoscedasticity (OLS third assumption), normality is assumed for big sample, following central limit theorem, the zero mean of the distribution of errors is accomplished due to inclusion of intercept in the regression, homoscedasticity assumption is generally hold, as we can see in the above graphs, it is handled due to log transformation of target variable, which is the most common fix for heteroscedasticity.

car_data_modified %>% 
  ggplot(aes(x = EngineV)) + 
  geom_histogram(fill = "dark orange") + 
  labs(
    title = "Histogram: Engine Volume",
    x = "Engine Volume",
    y = "Frequency"
  ) + my_ggtheme()


## After transformation, we can say that we got linear patterns in almost all plots now.

par(mfrow=c(2,2))

plot(car_data_modified$Mileage, car_data_modified$log_Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Mileage),
     ylim = range(car_data_modified$log_Price)
)

plot(car_data_modified$EngineV, car_data_modified$log_Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$EngineV),
     ylim = range(car_data_modified$log_Price)
)

plot(car_data_modified$Age, car_data_modified$log_Price,
     pch = 19,
     col = "darkgray",
     xlim = range(car_data_modified$Age),
     ylim = range(car_data_modified$log_Price)
)


## Linear regression

## We obtain RMSE of 0.299 under OLS and 0.3147 with 10-fold cross validation.

set.seed(48)
n <- nrow(car_data_modified)
train <- sample(1:n, n*0.8)

cars.train <- car_data_modified[train,]
cars.test <- car_data_modified[-train,]

lm.1 <- lm(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age, data = cars.train)
pred_lm1 <- lm.1 %>% predict(cars.test)
RMSE(pred_lm1, cars.test$log_Price)

regression.control <- trainControl(method = "cv", number = 10) # 10-fold cross validation
lm2 <- train(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age, 
             data = car_data_modified, method="lm", trControl = regression.control)
lm2


## Regression + Lasso

## Looks like the best model in terms of RMSE is a simple OLS model with no penalty (lamba = 0)

set.seed(48)
lambda_seq <- c(0:10, by = 0.25)

lasso1 <- train(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
                data = cars.train,
                preProcess = "scale",
                method = "glmnet",
                trControl = trainControl("cv", number = 10),
                tuneGrid = expand.grid(alpha = 1, lambda = lambda_seq)
)
plot(lasso1)

lasso1$bestTune$lambda


## KNN

## We obtain RMSE of 0.3447473 with knn.
## So looks like KNN does not perform better than our OLS and lasso regression model. 

knn_model <- train(
  log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
  data = cars.train,
  method = "knn",
  type = "anova", # tells knn this is regression
  trControl = trainControl("cv", number = 10),
  preProcess = c("center", "scale"),
  tuneLength = 50
)
plot(knn_model)

knn.predict <- knn_model %>% predict(cars.test)
RMSE(knn.predict, cars.test$log_Price)


## Boosting

## We obtain RMSE of 0.265289 with boosting, achieving better performance than OLS and 10-fold cross validation.

## create subset samples train, validation, test
set.seed(48)
n <- nrow(car_data_modified)
n1 = floor(n/2)
n2 = floor(n/4)
n3 = n-n1-n2
ii = sample(1:n,n)
carstrain =car_data_modified[ii[1:n1],]
carsval = car_data_modified[ii[n1+1:n2],]
carstest = car_data_modified[ii[n1+n2+1:n3],]
carstrainval = rbind(carstrain,carsval)

## finding optimal boosting parameters for validation set
treedepth = c(2, 4, 10)
treesnum = c(50, 500, 5000)
treelambda = c(.001, .2)
parmb = expand.grid(treedepth,treesnum,treelambda)
colnames(parmb) = c('tdepth','ntree','lam')
print(parmb)
nset = nrow(parmb)
olb = rep(0,nset)
ilb = rep(0,nset)
bfitv = vector('list',nset)

## create boosting parameters using values specified above 
for(i in 1:nset) {
  cat('doing boost ',i,' out of ',nset,'\n')
  boosttrain = gbm(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
                   data=carstrain, distribution='gaussian', interaction.depth=parmb[i,1], n.trees=parmb[i,2], shrinkage=parmb[i,3])
  ifit = predict(boosttrain, n.trees=parmb[i,2])
  ofit = predict(boosttrain, newdata=carsval, n.trees=parmb[i,2])
  olb[i] = sum((carsval$log_Price-ofit)^2)
  ilb[i] = sum((carstrain$log_Price-ifit)^2)
  bfitv[[i]] = boosttrain
}

## compute for rmse in vs out of sample
ilb = round(sqrt(ilb/nrow(carstrain)),3) 
olb = round(sqrt(olb/nrow(carsval)),3)

## print rmse values and find where it is at lowest
print(cbind(parmb, olb, ilb))
which.min(olb)

## which variables have the most influence
vimportance = summary(boosttrain)

# use best fit on test
boosttest = gbm(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
                data=carstrainval,distribution='gaussian', interaction.depth=10,n.trees=50,shrinkage=.2)
boosttestpred=predict(boosttest,newdata=carstest,n.trees=50)
boosttestrmse = sqrt(sum((carstest$log_Price-boosttestpred)^2)/nrow(carstest))
print(boosttestrmse)

# plot actual vs predicted values from boosting model
plot(carstest$log_Price,boosttestpred,xlab='Test Price',ylab='Boosting Prediction')
abline(0,1, col='red')
vimportance=summary(boosttest)


## Regression Tree

## We obtain RMSE of 0.3897077 with regression tree.

# original big tree
temp = tree(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
            data = cars.train, mindev=.0001)
summary(temp)
plot(temp)
text(temp)

## cross validation, simplest model with lowest deviation = 12
cv_car = cv.tree(temp)
plot(cv_car$size, cv_car$dev, type='b')

## pruning
prune_car = prune.tree(temp, best=12)
plot(prune_car)
text(prune_car)

## prediction
yhat = predict(prune_car, newdata = car_data_modified[-train,])

car.test = car_data_modified[-train, 'log_Price']
plot(yhat,car.test)
abline(0,1)

sqrt(mean((yhat-car.test)^2))


## Bagging

## We obtain RMSE of 0.3428916 with bagging, better than simple regression tree.

## Set up model
bag_car = bagging(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
                  data = cars.train, coob = TRUE)
print(bag_car)

## prediction
yhat_bag = predict(bag_car, newdata = car_data_modified[-train,])
car_bag.test = car_data_modified[-train, 'log_Price']
plot(yhat_bag,car_bag.test)
abline(0,1, col='red')
sqrt(mean((yhat_bag-car_bag.test)^2))

## Visualize the Importance of the Predictors

# calculate variable importance
df_temp <- car_data_modified[,-2]
VI <- data.frame(var=names(df_temp[,-8]), imp=varImp(bag_car))

# sort variable importance descending
VI_plot <- VI[order(VI$Overall, decreasing=TRUE),]

#visualize variable importance with horizontal bar plot
barplot(VI_plot$Overall,
        names.arg=rownames(VI_plot),
        horiz=TRUE,
        col='steelblue',
        xlab='Variable Importance')


## Random Forest

## We obtain RMSE of 0.2373476 with random forest.

## create subset samples train, validation, test
finrf = randomForest(log_Price ~ Brand + Body + Mileage + EngineV + Engine.Type + Registration + Age,
                     data=cars.train)
finrf

# Getting Convergence at tree size = 100
par(mfrow=c(1,1))
plot(finrf)

## prediction
yhat_rf = predict(finrf, newdata = car_data_modified[-train,])
car_rf.test = car_data_modified[-train, 'log_Price']
plot(yhat_rf,car_rf.test)
abline(0,1, col='red')
sqrt(mean((yhat_rf-car_rf.test)^2))

## Visualize the Importance of the Predictors
varImpPlot(finrf)
importance(finrf)

