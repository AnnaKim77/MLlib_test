
## iris data 내장, kmeans 기본 제


data(iris)

# iris dataset 형태 확인
head(iris)
str(iris)

# Clustering
set.seed(20160823)
iris.kmeans<-kmeans(iris[,1:4],3)
iris.kmeans
table(iris.kmeans$cluster,iris$Species)

# Visualization
library(ggplot2)
library(sqldf)

iris$cluster<-iris.kmeans$cluster
iris$Species<-as.integer((iris$Species))
sqldf("select Species, cluster, count(*) from iris group by Species, cluster")
mis.point <-sqldf("select * from iris where (Species!=cluster)")
p<-ggplot(iris,aes(Petal.Length,Petal.Width,color=factor(cluster)))+geom_point()+labs(colour="Cluster")
p + geom_point(data=mis.point, aes(x=Petal.Length, y=Petal.Width, shape=factor(cluster), alpha=.6, size=4.5,colour="red"),show.legend=FALSE)