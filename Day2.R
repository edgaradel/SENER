
library(tidyverse)

1/sqrt(2*pi)*exp(-1/2) #by hand
dnorm(x = 1, mean = 0, sd = 1) #use built-in density function
dnorm(x = 0, mean = 0, sd = 1)

pnorm(1, mean = 0, sd = 1)

pnorm(1) - pnorm(-1)

pnorm(2) - pnorm(-2)

qnorm(c(.025,.975))

ggplot(data.frame(x=c(-4,4)),aes(x))+stat_function(fun=dnorm)+ylab("p(x | 0,1)")+stat_function(fun=dnorm,xlim=c(-4,0),geom='area')+ylab("")+annotate("text",-1,.1,label="?",color="white")


pnorm(1, mean = 0, sd = 1) #pnorm(b) gives p(x < b)


ggplot(data.frame(x=c(-4,4)),aes(x))+stat_function(fun=dnorm)+
  stat_function(fun=dnorm,xlim=c(-1,1),geom='area')+ylab("")+
  annotate("text",0,.15,label="?",color="white")


#pnorm(b)-pnorm(a) gives p(a<x<b)
pnorm(1) - pnorm(-1) #note that mean=0, sd=1 are always default

qnorm(.95) #mean=0 and sd=1 by default

qnorm(c(.025,.975)) #middle 95% between 2.5 and .975 percentile


## -------------------------------------------------------------------------------------------------------------------
df<-read.csv("http://www.nathanielwoodward.com/naive.csv")
df%>%group_by(Class)%>%summarize_all(.funs=list(mean="mean",sd="sd"))%>%select(1,2,5,3,6,4,7)%>%as.data.frame


#manually
1/sqrt(2*pi*.187^2)*exp(-(6-5.855)^2 / (2*.187^2))
#or use the built-in density calculator
dnorm(x=6, mean=5.855, sd=.187)



#naive bayes in R
params <- df%>%group_by(Class)%>%summarize_all(.funs=list(mean="mean",sd="sd"))
Female <- params[1,] #means and sds for F
Male <- params[2,] #means and sds for M

new<-data.frame(Height=6, Weight=130, Shoe=8) #new observation

probM=prod( #compute M numerator: p(M)*P(Height | M)*P(Weight | M)*P(Shoe | M)
  .5,
  dnorm(new$Height, Male$Height_mean, Male$Height_sd),
  dnorm(new$Weight, Male$Weight_mean, Male$Weight_sd),
  dnorm(new$Shoe,   Male$Shoe_mean,   Male$Shoe_sd)
)
probF=prod( #compute F numerator: p(F)*P(Height | F)*P(Weight | F)*P(Shoe | F)
  .5,
  dnorm(new$Height, Female$Height_mean, Female$Height_sd),
  dnorm(new$Weight, Female$Weight_mean, Female$Weight_sd),
  dnorm(new$Shoe,   Female$Shoe_mean,   Female$Shoe_sd)
)

c(probM,probF) #which is bigger?
c(probM,probF)/sum(c(probM,probF)) #normalize (divide by the sum): same answer


#####
install.packages(naivebayes)
library(naivebayes) #use someone else's function!
#First, "train" the classifier (i.e., estimate means/sds for each class)
nb_classifier<-gaussian_naive_bayes(df[,2:4], df[,1], prior=c(.5,.5))
nb_classifier$params

## Then, use these estimates to predict class for new observation
as.matrix(new) #new data
predict(nb_classifier,newdata=as.matrix(new),type="prob") #predicted probs
predict(nb_classifier,newdata=as.matrix(new),type="class") #predictied class



############################

#possible values of theta
Theta <- c(.25,.5,.75)
#probability of each theta before seeing data
Prior <- c(.25,.5,.25) 
#observe 1 heads
y <- 1 
#probability of each theta given a heads
Likelihood <- Theta^y*(1-Theta)^(1-y)
#prior times likelihood is unnormalized posterior
Prior_x_Likelihood <- Prior * Likelihood 
#normalized posterior (sums to 1)
Posterior <- Prior_x_Likelihood / sum( Prior_x_Likelihood) 

data.frame(Theta, Prior, Likelihood, Prior_x_Likelihood, Posterior)

sum(Theta*Posterior) #posterior mean
Theta[which.max(Posterior)] #posterior mode


Prior <- Posterior #posterior as new prior
Posterior <- Prior*Likelihood / sum( Prior*Likelihood ) 
data.frame(Theta,Prior,Likelihood,Posterior)


Prior <- Posterior #make posterior the new prior
Likelihood <- Likelihood^3 #likelihood of 3 more heads in a row
Posterior <- Prior*Likelihood / sum( Prior*Likelihood)

data.frame(Theta,Prior,Likelihood,Posterior)

Prior <- Posterior
Likelihood <- Theta^0*(1-Theta)^(1-0)
Posterior <- Prior*Likelihood / sum( Prior*Likelihood)

data.frame(Theta,Prior,Likelihood,Posterior)



#####

curve(x^5*(1-x)^1)

Theta <- c(.25,.5, .75)
Prior <- c(.25,.5,.25)
Likelihood <- Theta^5*(1-Theta)^1
Posterior <- Prior*Likelihood / sum(Prior*Likelihood)

data.frame(Prior, Likelihood, Posterior)



#########

Theta <- seq(0,1,.05) #21 evenly spaced values between 0 and 1
Prior <- 1 
Likelihood <- Theta^5*(1-Theta)^1
Posterior <- Prior*Likelihood / sum( Prior*Likelihood ) 
head(data.frame(Theta, Prior, Likelihood, Posterior))


sum(Theta*Prior) #prior mean
sum(Theta*Posterior) #posterior mean
Theta[which.max(Posterior)] #posterior mode


## -------------------------------------------------------------------------------------------------------------------
ggplot()+stat_function(aes(x=c(0,1)),fun=dbeta,args=list(6,2),geom="line")

#mean 
6/(6+2)
#mode (MAP)
5/(6+2-2)
#variance
(6*2)/((6+1)^2*(6+2+1))


## -------------------------------------------------------------------------------------------------------------------
#95% bayesian credible interval (equal-tailed)
qbeta(c(.025,.975),6,2)

curve(dbeta(x,6,2)) 

1-pbeta(.5,6,3)

pbeta(.65,6,3)-pbeta(.55,6,3)


qbeta(c(.025,.975),6,3)



#### RUN THIS
hdi=function(df, x.min, x.max, prob=.95, ...) {
  p <- function(h) { g <- function(x) {y <- df(x); ifelse(y > h, y, 0)}
  integrate(g, x.min, x.max, ...)$value - prob }
  y<- uniroot(p, c(x.min, x.max), tol=1e-12)$root #y intercept
  mid=optimize(df,c(x.min,x.max),maximum=T, tol=1e-12)$maximum #  mode
  if(abs(x.min-mid) >.0001 & abs(x.max-mid) >.0001){
  c(uniroot(function(x)df(x)-y,lower=x.min,upper=mid, tol=1e-12)$root,
         uniroot(function(x)df(x)-y,lower=mid,upper=x.max, tol=1e-12)$root)}
  else if(x.min-mid<0){c(0,uniroot(function(x)df(x)-y,lower=x.min,upper=x.max, tol=1e-12)$root)}
  else c(uniroot(function(x)df(x)-y,lower=x.max,upper=x.min, tol=1e-12)$root,0)
}
hdi(function(x)dbeta(x,7,1),0,1)

qbeta(c(.025,.975),1,11)

hdi(function(x)dbeta(x,7,1),0,1)

par(mfrow=c(1,2))
curve(dbeta(x,1,1),0,1)
curve(dbeta(x,.5,7),0,1)

pbeta(.5,.5,7) #prob that true proportion less than .5 for informative prior
pbeta(.5,1,1) #prob that true proportion less than .5 for uninformative prior

## -------------------------------------------------------------------------------------------------------------------
batting<-read.csv("../data/batting.csv")
bat<-batting%>%filter(AB>500)

ggplot(bat,aes(average))+geom_histogram()

mean1=mean(bat$average) #empirical mean batting average
var1=var(bat$average) #empirical sd batting average

alpha <- ((1 - mean1) / var1 - 1 / mean1) * mean1 ^ 2
beta <- alpha * (1 / mean1 - 1)

#MASS::fitdistr(bat$average,densfun = dbeta,start=list(shape1=70,shape2=200))

ggplot(bat,aes(x=average))+geom_histogram(aes(y=..density..))+stat_function(fun=function(x)dbeta(x,80,229),geom="line")


## -------------------------------------------------------------------------------------------------------------------
qbeta(c(.025,.975),92,245) #Vic Rodriguez
qbeta(c(.025,.975),3851,8822) #Hank Aaron


## -------------------------------------------------------------------------------------------------------------------
aaron.samp <- rbeta(10000, 80+3371, 229+(12364-3771))
rodriquez.samp <- rbeta(1000, 80+12, 229+(28-12))

mean(aaron.samp>rodriquez.samp)


## ----echo=T---------------------------------------------------------------------------------------------------------
batting<-batting%>%mutate(estimate=(80+H)/(80+229+AB))


## ----fig.height=3---------------------------------------------------------------------------------------------------
batting%>%mutate(group=factor(ntile(AB,4)))%>%
  ggplot(aes(average,estimate,color=AB))+geom_point()+geom_abline(slope=1,intercept=0)+
  scale_color_gradient( trans = "log", breaks = 10^(1:4) )+
  geom_hline(yintercept=80/(80+229),color="red",lty=2)
