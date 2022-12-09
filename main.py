import tkinter as tk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import sys
from Hopfield import Hopfield

def trigger_step():
    global ans_d, test_d
    test_d = models[dataset.get()].step(test_d)
    draw_picture()


def trigger_select_test(select):
    global ans_d, test_d, options
    if select == options[0]: # noisy data
        ans_d, test_d = models[dataset.get()].get_noise_data()
    else: # test data
        select_test = select.split()[1]
        ans_d = models[dataset.get()]._xs[int(select_test)-1]
        test_d = models[dataset.get()]._ys[int(select_test)-1]
    draw_picture()


def draw_picture():
    global ans_d, test_d
    test_ax.imshow(models[dataset.get()].print_d(test_d), cmap='Oranges', resample=True, aspect='equal')
    ans_ax.imshow(models[dataset.get()].print_d(ans_d), cmap='Oranges', resample=True, aspect='equal')
    canvas.draw()
    canvas.flush_events()


def trigger_update_option():
    global options
    options = options[0:1] + [f"Class {i+1}" for i in range(len(models[dataset.get()]._xs))] # new options for OptionMenu(dropdown)
    menu['menu'].delete(0, tk.END)
    for select in options:
        menu['menu'].add_command(label=select, command=tk._setit(option, select, trigger_select_test))
    option.set(options[0])
    trigger_select_test(options[0])


models = {"Basic": Hopfield(9, 12).train("Basic"), "Bonus": Hopfield(10, 10).train("Bonus")}
ans_d, test_d = models["Basic"].get_noise_data()

root = tk.Tk()
root.title("Hopfield")
root.protocol("WM_DELETE_WINDOW", lambda: sys.exit(0))


# Dataset
dataset = tk.StringVar(root, value="Basic")
dataset_label = tk.Label(root, text="Dataset:")
dataset_label.grid(row=1, column=0)
basic_button = tk.Radiobutton(root, text="Basic", value="Basic", variable=dataset, command=trigger_update_option)
basic_button.grid(row=1, column=1)
bonus_button = tk.Radiobutton(root, text="Bonus", value="Bonus", variable=dataset, command=trigger_update_option)
bonus_button.grid(row=1, column=2)
basic_button.select()

# Picture
fig = Figure(figsize=(4, 4), dpi=100)
canvas = FigureCanvasTkAgg(fig, master=root)
test_ax = fig.add_subplot(121)
ans_ax = fig.add_subplot(122)
test_ax.set_title("Test Step:")
test_ax.get_xaxis().set_visible(False)
test_ax.get_yaxis().set_visible(False)
ans_ax.set_title("Answer:")
ans_ax.get_xaxis().set_visible(False)
ans_ax.get_yaxis().set_visible(False)
draw_picture()
canvas.get_tk_widget().grid(row=0, column=0, columnspan=3)

# Select Test
options = ["Random Noise"]
option = tk.StringVar(root, value=options[0])
option_label = tk.Label(root, text="Test Data:")
option_label.grid(row=2, column=0)
menu = tk.OptionMenu(root, option, *options, command=trigger_select_test)
menu.grid(row=2, column=1, columnspan=2)
trigger_update_option()

# Step Button
btn = tk.Button(root, text="Next", command=trigger_step)
btn.grid(row=3, column=0, columnspan=3)

root.mainloop()
