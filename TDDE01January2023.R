set.seed(1234)

# produce the training data in dat

x <- runif(500,-4,4)
y <- sin(x)
dat <- cbind(x,y)
plot(dat)

gamma <- 0.01

h <- function(z){
  
  # activation function (sigmoid)
  
  return(1/(1+exp(-z)))
}

hprime <- function(z){
  
  # derivative of the activation function (sigmoid)
  
  return(h(z) * (1 - h(z)))
}

yhat <- function(x){

  # prediction for point x
  
q0 <- x
z1 <- w1 %*% q0 + b1
q1 <- as.matrix(apply(z1,1,h), nrow = 2, ncol = 1)
z2 <- w2 %*% q1 + b2
return(z2)
}

MSE <- function(){
  
  # mean squared error
  
  res <- NULL
  for(i in 1:nrow(dat)){
    res <- c(res,(dat[i,2] - yhat(dat[i,1])) ^ 2)
    }
  return(mean(res))
}

# initialize parameters

w1 <- matrix(runif(2,-.1,.1), nrow = 2, ncol = 1)
b1 <- matrix(runif(2,-.1,.1), nrow = 2, ncol = 1)

w2 <- matrix(runif(2,-.1,.1), nrow = 1, ncol = 2)
b2 <- matrix(runif(1,-.1,.1), nrow = 1, ncol = 1)

res <- NULL
for(i in 1:100000){
  if(i %% 1000 == 0){
    res <- c(res,MSE())
  }
  
  # forward propagation
  
  j <- sample(1:nrow(dat),1)
  q0 <- dat[j,1]
  z1 <- w1 %*% q0 + b1
  q1 <- as.matrix(apply(z1,1,h), nrow = 2, ncol = 1)
  z2 <- w2 %*% q1 + b2
  
  # backward propagation
  
  dz2 <- - 2 * (dat[j,2] - z2)
  dq1 <- t(w2) %*% dz2
  dz1 <- dq1 * hprime(z1)
  dw2 <- dz2 %*% t(q1)
  db2 <- dz2
  dw1 <- dz1 %*% t(q0)
  db1 <- dz1
  
  # parameter updating
  
  w2 <- w2 - gamma * dw2
  b2 <- b2 - gamma * db2
  w1 <- w1 - gamma * dw1
  b1 <- b1 - gamma * db1
}
plot(res, type = "l")

plot(dat)
points(dat[,1],lapply(dat[,1],yhat),col="red")
