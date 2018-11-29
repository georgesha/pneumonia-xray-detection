import matplotlib.pyplot as plt

file = open("cnn3.log")
lines = file.readlines()
loss = []
acc = []
for line in lines:
    line_elements = line.split(" ")
    if "loss:" in line_elements:
        loss.append(float(line_elements[line_elements.index("loss:") + 1]))
        acc.append(float(line_elements[line_elements.index("acc:") + 1]))
        if len(loss) == 4000:
            break

plt.plot(range(4000), loss, label="loss")
plt.savefig("loss.png")
plt.clf()

plt.plot(range(4000), acc, label="accuracy")
plt.savefig("accuracy.png")
