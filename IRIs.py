from sklearn.datasets import load_iris
iris = load_iris()
type(iris)
print(iris.data)
print(iris.feature_names)
print(iris.target)#also called response
print(iris.target_names)
#classification is supervised, response is categorical
#regression is supervised, response is ordered and continuos
print(iris.data.shape)
