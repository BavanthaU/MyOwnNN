from Value import Value
from Visualiser import draw_dot
from MLP import MLP

xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0]
]

ys = [1.0, -1.0, -1.0, 1.0]

n = MLP(3, [4, 4, 1])

# gradient decent
for i in range(1000):
    ypred = [n(x) for x in xs]
    print(ypred)

    # how is it performing = calculate the loss
    loss = sum([(yout - ygt) ** 2 for ygt, yout in zip(ys, ypred)])
    print(loss)

    # need to reduce the loss --> lets go backwards and tune weight by small amounts towards reducing loss
    # backward pass
    for p in n.parameters():
        p.grad = 0.0
    loss.backward()
    dot = draw_dot(loss)
    # dot.render(directory='digraph_output', view=True)

    for p in n.parameters():
        p.data += -0.01 * p.grad  # "-" is to decrease the loss where 0.01 is the learning rate
