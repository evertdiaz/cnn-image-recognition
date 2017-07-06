# Modulo de Tensorflow
import tensorflow as tf
# Tensores
# Rank: Numero de Dimensiones
# Shape: Array de Dimensiones

node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node4 = tf.constant([5.0])
minodo = tf.constant([
  [[3.0,2.8,2.0],[1.0,2.0,3.0],[2.9,9.1,0.8]],
  [[1.5,2.5,3.0],[1.9,1.8,9.0],[2.0,3.0,9.0]]
  ])
print(node1, node2, node4, minodo)

# La Sesion para correr modelos de Tensorflow
sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print('node3: ', node3)
print('sess.run(node3: ', sess.run(node3))

# Placeholders (objetos anonimas)
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b # + es igual a tf.add(a,b)
print(sess.run(adder_node, {a: 3, b:4.5}))
print(sess.run(adder_node, {a: [1,3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a:3,b:4.5}))

# Uso de Variables
w = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = w * x + b

# En tensorflow se debe inicializar las variables
init = tf.global_variables_initializer() 
sess.run(init)
print(sess.run(linear_model, {x:[1,2,3,4]}))

y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Cambiar valor de variables
fixw = tf.assign(w, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixw, fixb])
print(sess.run(loss, {x:[1,2,3,4], y:[0,-1,-2,-3]}))

# Entrenando ejemplo
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init)
print 'Antes de Entrenar: ',sess.run([w,b])
for i in range(1000):
  sess.run(train, {x:[1,2,3,4], y:[0,-1,-2,-3]})
print 'Despues de Entrenar: ', sess.run([w,b])

# Ahora usando aprendizaje contrib.learn
import numpy as np
features = [tf.contrib.layers.real_valued_column('x', dimension=1)]
estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_train}, y_train, batch_size=4, num_epochs=1000)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn({'x':x_eval}, y_eval, batch_size=4, num_epochs=1000)
estimator.fit(input_fn=input_fn, steps=1000)
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print('error de entrenamiento: %r'% train_loss)
print('error en test: %r'% eval_loss)
