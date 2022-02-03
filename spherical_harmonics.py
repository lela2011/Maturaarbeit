import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spec
from matplotlib.widgets import Button

if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    theta, phi = np.meshgrid(np.linspace(0, 2 * np.pi, 100), np.linspace(0, np.pi, 100))
    R = abs(spec.sph_harm(0, 0, theta, phi).real)

    x = R * np.sin(phi) * np.cos(theta)
    y = R * np.sin(phi) * np.sin(theta)
    z = R * np.cos(phi)

    class Buttons:

        l = 0
        m = 0

        def next_l(self, event):
            self.l += 1
            self.new_data()

        def prev_l(self, event):
            if not self.l==0:
                self.l -= 1
                self.new_data()

        def next_m(self, event):
            if not self.m == self.l:
                self.m += 1
                self.new_data()

        def prev_m(self, event):
            if not self.m == -self.l:
                self.m -= 1
                self.new_data()

        def new_data(self):
            global R, x, y, z, theta, phi, ax
            if self.m == 0:
                R = abs(spec.sph_harm(self.m, self.l, theta, phi).real)
            else:
                R = abs(spec.sph_harm(self.m, self.l, theta, phi).imag)

            x = R * np.sin(phi) * np.cos(theta)
            y = R * np.sin(phi) * np.sin(theta)
            z = R * np.cos(phi)

            ax.cla()

            ax.plot_surface(x, y, z)
            ax.set_title(f"l: {self.l}\nm: {self.m}")
            ax.set_axis_off()
            plt.plot()

    callback = Buttons()

    axes_1 = plt.axes([0.000001, 0.075001, 0.15, 0.075])
    bnext_l = Button(axes_1, 'Next l')
    bnext_l.on_clicked(callback.next_l)

    axes_2 = plt.axes([0.000001, 0.000001, 0.15, 0.075])
    bprev_l = Button(axes_2, 'Previous l')
    bprev_l.on_clicked(callback.prev_l)

    axes_3 = plt.axes([0.15001, 0.075001, 0.15, 0.075])
    bnext_m = Button(axes_3, 'Next m')
    bnext_m.on_clicked(callback.next_m)

    axes_4 = plt.axes([0.150001, 0.000001, 0.15, 0.075])
    bprev_m = Button(axes_4, 'Previous m')
    bprev_m.on_clicked(callback.prev_m)

    ax.plot_surface(x,y,z)
    ax.set_title("l: 0\nm: 0")
    ax.set_axis_off()

    plt.show()