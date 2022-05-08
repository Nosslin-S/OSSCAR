import numpy as np
import os

import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from ase import Atoms
from ase.io.trajectory import Trajectory
from sympy import *
from NGLUtilsClass import NGLWidgets

class NGLTrajectory(NGLWidgets):
    def __init__(self, trajectory):
        super().__init__(trajectory)
        
        self.output_ratio=widgets.Output()

        
        self.slider_amplitude = widgets.FloatSlider(
            value=0.06,
            min=0.01,
            max=0.12,
            step=0.01,
            # description="Amplitude",
            continuous_update=False,
            layout=widgets.Layout(width="200px")
        )
        self.slider_amplitude_description=widgets.HTMLMath(r"Amplitude",layout=self.layout_description)

        self.slider_M = widgets.FloatSlider(
            value=2,
            min=1,
            max=5,
            step=0.1,
            continuous_update=False,
            layout=widgets.Layout(width="200px")
        )
        self.slider_M_description=widgets.HTMLMath(r"$\frac{\text{M}_{\text{red}}}{\text{M}_{\text{grey}}}$",layout=self.layout_description)

        self.slider_C=widgets.FloatSlider(
            value=1,
            min=0.5,
            max=5,
            step=0.1,
            description="Force constant",
            continuous_update=False,
        )

        self.button_chain=widgets.RadioButtons(
            options=['monoatomic', 'diatomic'],
            value='monoatomic',
            # description='Atomic chain type: ',
            disabled=False
        )
        self.button_chain_description = widgets.HTMLMath(r"Atomic chain type",layout=self.layout_description)
        
        self.slider_amplitude.observe(self.compute_trajectory_1D, "value")
        self.slider_M.observe(self.compute_trajectory_1D, "value")
        self.slider_M.observe(self.update_band_M, "value")
        # self.slider_C.observe()
        self.button_chain.observe(self.compute_dispersion, "value")
        self.button_chain.observe(self.compute_trajectory_1D, "value")
        self.button_chain.observe(self.band_dispersion, "value")
        self.button_chain.observe(self.show_slider_M,"value")
        plt.ioff()
        px = 1 / plt.rcParams["figure.dpi"]
        self.fig, self.ax = plt.subplots(figsize=(500 * px, 300 * px))

        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        (self.lines_ac,) = self.ax.plot([], [], c="blue")
        (self.lines_op,) = self.ax.plot([], [], c="orange")
        (self.lines,) = self.ax.plot([], [], c="blue")
        (self.lines_ac_out,) = self.ax.plot([], [],'--', c="blue",alpha=0.5)
        (self.lines_op_out,) = self.ax.plot([], [],'--', c="orange",alpha=0.5)
        (self.lines_out,) = self.ax.plot([], [],'--', c="blue",alpha=0.5)
        (self.point,) = self.ax.plot([], [], ".", c="crimson", markersize=15)
        self.ax.set_xlabel("k")
        self.ax.set_ylabel("$\omega$")
        self.fig.set_tight_layout(tight=True)
        plt.ion()

        self.optic = False
        self.ka = 0
        self.ka_array = np.linspace(-2*np.pi, 2*np.pi, 101)
        self.idx = 50
        self.a=1
        self.x=0
        self.y=0
        # self.compute_dispersion()

        # self.compute_trajectory_1D()
        # self.band_dispersion()

        # self.atoms = widgets.HBox(
        #     [widgets.VBox([self.slider_amplitude, self.slider_M]), self.fig.canvas]
        # )
        self.init_delay=20
    def compute_dispersion(self,*args):
        if self.button_chain.value == 'monoatomic':
            a = Symbol("a")
            a1 = Matrix([a])
            k = symbols("k")
            u, w = symbols("u w")
            M1, C = symbols("M1 C")

            atom_positions = Matrix([[-1], [1]])
            RHS = 0 * a
            for i in range(atom_positions.rows):
                position = atom_positions.row(i)
                l = position
                K = Matrix([l * k])
                RHS += u * exp(I * (K.T.dot(a1))) - u

            RHS *= C 
            RHS= RHS.rewrite(cos).simplify()
            LHS = -M1*w**2*u
            Eq=RHS-LHS
            sols=solve(Eq, w)
            Sol=sols[0]
            self.w_ = lambdify((k, C), Sol.subs({ M1: 1, a: 1}))

            self.compute_dispersion_relation()

        elif self.button_chain.value == 'diatomic':
            a = Symbol("a")
            a1 = Matrix([a])
            k = symbols("k")
            u, v = symbols("u v")
            M1, M2, C = symbols("M1 M2 C")

            atom_positions_u = Matrix([[0], [1]])
            atom_positions_v = Matrix([[-1], [0]])
            RHS1 = 0 * a
            RHS2 = 0 * a
            for i in range(atom_positions_v.rows):
                position = atom_positions_v.row(i)
                l = position
                K = Matrix([l * k])
                RHS1 += v * exp(I * (K.T.dot(a1))) - u

            for i in range(atom_positions_u.rows):
                position = atom_positions_u.row(i)
                l = position
                K = Matrix([l * k])
                RHS2 += u * exp(I * (K.T.dot(a1))) - v

            RHS1 *= -C / M1
            RHS2 *= -C / M2

            matrix = linear_eq_to_matrix([RHS1, RHS2], [u, v])[0]
            matrix.simplify()
            eig1, eig2 = matrix.eigenvects()

            self.A_1 = lambdify(
                (k, M2, C, a), eig1[2][0][0].subs({ M1: 1})
            )
            self.w2_1 = lambdify((k, M2, C, a), eig1[0].subs({ M1: 1}))

            self.A_2 = lambdify(
                (k, M2, C, a), eig2[2][0][0].subs({ M1: 1})
            )
            self.w2_2 = lambdify((k, M2, C, a), eig2[0].subs({ M1: 1}))
            self.compute_dispersion_relation()

    def compute_dispersion_relation(self, *args):
        if self.button_chain.value == 'monoatomic':
            self.w = self.w_(self.ka_array, self.slider_C.value)

        elif self.button_chain.value == 'diatomic':
            A_1 = self.A_1(self.ka_array, self.slider_M.value,self.slider_C.value, 1)
            A_2 = self.A_2(self.ka_array, self.slider_M.value,self.slider_C.value, 1)

            w_1 = np.sqrt(self.w2_1(self.ka_array, self.slider_M.value,self.slider_C.value, 1))
            w_2 = np.sqrt(self.w2_2(self.ka_array, self.slider_M.value,self.slider_C.value, 1))

            # Move all the highest frequencies in w_2 (optical branch)
            for i in range(len(w_1)):
                if w_1[i] > w_2[i]:
                    w_1[i], w_2[i], A_1[i], A_2[i] = w_2[i], w_1[i], A_2[i], A_1[i]

            self.w_ac = w_1
            self.w_op = w_2
            self.A_ac = A_1
            self.A_op = A_2

    def compute_trajectory_1D(self, *args):
        # tmpdirname=self.tmpdir.name
        tmpdirname='.'
        filename=os.path.join(tmpdirname,"trajectory_1d.traj")
        self.ka = self.ka_array[self.idx]
        traj=Trajectory(filename, "w")
        if self.button_chain.value == 'monoatomic':
            ax = np.array([1, 0, 0])
            w=self.w[self.idx]
            atom_positions = []
            K = np.array([self.ka, 0, 0])
            

            for frame in np.linspace(0, 50, 50):
                atom_positions = []
                if w != 0:
                    t = 2 * np.pi / 50 / np.real(w) * frame
                else:
                    t = 0
                for i in range(20):
                    atom_positions_ = (
                        -5 * ax
                        + i * ax*0.5
                        + np.real(
                            self.slider_amplitude.value
                            * np.exp(1j * w * t)
                            * np.exp(1j * i * np.dot(K, ax))
                        )
                        * ax
                    )
                    atom_positions.append(atom_positions_)

                atoms = Atoms(int(len(atom_positions)) * "C", positions=atom_positions)
                traj.write(atoms)

        elif self.button_chain.value == 'diatomic':
            if self.optic:
                self.w = self.w_op[self.idx]
                self.A = self.A_op[self.idx]
            else:
                self.w = self.w_ac[self.idx]
                self.A = self.A_ac[self.idx]

            ax = np.array([1, 0, 0])

            # normalize the relative amplitudes so that the positions don't explode if one atom static
            # nums1 = [self.A / np.absolute(self.A), self.A]
            # nums2 = [1 / np.absolute(self.A), 1]
            # idx_min=np.abs(nums1).argmin()
            # amp1 = nums1[idx_min]
            # amp2 = nums2[idx_min]
            amp_vector=np.array([self.A,1])
            amp_vector=amp_vector/np.linalg.norm(amp_vector)
            amp1,amp2=amp_vector[0],amp_vector[1]
            atom_positions = []
            K = np.array([self.ka, 0, 0])
        

            for frame in np.linspace(0, 50, 50):
                atom_positions = []
                if self.w != 0:
                    t = 2 * np.pi / 50 / np.real(self.w) * frame
                else:
                    t = 0
                for i in range(10):
                    atom_positions_1 = (
                        -5 * ax
                        + i * ax
                        + np.real(
                            self.slider_amplitude.value
                            * amp1
                            * np.exp(1j * self.w * t)
                            * np.exp(1j * i * np.dot(K, ax))
                        )
                        * ax
                    )
                    atom_positions_2 = (
                        -5 * ax
                        + (1 / 2 * ax + i * ax)
                        + np.real(
                            self.slider_amplitude.value
                            * amp2
                            * np.exp(1j * self.w * t)
                            * np.exp(1j * i * np.dot(K, ax))
                        )
                        * ax
                    )
                    atom_positions.append(atom_positions_1)
                    atom_positions.append(atom_positions_2)

                atoms = Atoms(int(len(atom_positions) / 2) * "CO", positions=atom_positions)
                traj.write(atoms)
        new_traj=Trajectory(filename)
        self.replace_trajectory(
                    traj = new_traj, representation="spacefill"
                )
        # time.sleep(0.1)
        # tmpdir.cleanup()
        self.view.control.zoom(0.25)

    def band_dispersion(self,*args):
        px = 1 / plt.rcParams["figure.dpi"]
        plt.ioff()
        
        if self.button_chain.value == 'monoatomic':
            self.fig.set_figwidth(500*px)
            self.ax.set_xticks(np.linspace(-np.pi,np.pi,5))
            self.ax.set_xticklabels([r'$-\frac{\pi}{a}$','','0','',r'$\frac{\pi}{a}$'])
            self.lines_out.set_data((self.ka_array,self.w))
            self.lines.set_data((self.ka_array[25:75],self.w[25:75]))
            # self.lines_out.set_data((np.append(self.ka_array[:25],self.ka_array[75:]),np.append(self.w[:25],self.w[75:])))
            
            self.lines_ac.set_data(([], []))
            self.lines_op.set_data(([], []))
            self.lines_ac_out.set_data(([], []))
            self.lines_op_out.set_data(([], []))

        elif self.button_chain.value == 'diatomic':
            self.fig.set_figwidth(350*px)
            self.ax.set_xticks(np.linspace(-np.pi,np.pi,5))
            self.ax.set_xticklabels([r'$-\frac{\pi}{2a}$','','0','',r'$\frac{\pi}{2a}$'])
            self.ax.set_ylabel('$\omega$')
            self.lines_ac_out.set_data((self.ka_array,self.w_ac))
            self.lines_op_out.set_data((self.ka_array,self.w_op))
            self.lines_ac.set_data((self.ka_array[25:75],self.w_ac[25:75]))
            self.lines_op.set_data((self.ka_array[25:75],self.w_op[25:75]))
            # self.lines_ac.set_data((np.append(self.ka_array[:25],self.ka_array[75:]),np.append(self.w_ac[:25],self.w_ac[75:])))
            # self.lines_op.set_data((np.append(self.ka_array[:25],self.ka_array[75:]),np.append(self.w_op[:25],self.w_op[75:])))
            self.lines.set_data(([], []))
            self.lines_out.set_data(([], []))

        self.ax.plot([-np.pi,-np.pi],[0,2.2],'k--',linewidth=1)
        self.ax.plot([np.pi,np.pi],[0,2.2],'k--',linewidth=1)

        self.ax.set_xbound((-2*np.pi) / self.a, (2*np.pi) / self.a)
        self.ax.set_ybound(0, 2.2)


        self.point.set_data((0, 0))
        self.fig.canvas.mpl_connect("button_press_event", self.onclick)
        plt.ion()

    def update_band_M(self, *args):
        self.compute_dispersion_relation()
        # Re-adjust frequency to closest one
        self.idx = (np.abs(self.ka_array - self.x)).argmin()
        self.ka = self.ka_array[self.idx]
        w = self.compute_distance(y=self.y)
        self.point.set_data(self.ka, w)
        self.compute_trajectory_1D()
        # Update band dispersion lines
        self.lines_ac.set_data((self.ka_array[25:75], self.w_ac[25:75]))
        self.lines_op.set_data((self.ka_array[25:75], self.w_op[25:75]))
        self.lines_ac_out.set_data((self.ka_array,self.w_ac))
        self.lines_op_out.set_data((self.ka_array,self.w_op))

    def onclick(self, event):
        self.x = event.xdata
        self.y = event.ydata

        self.idx = (np.abs(self.ka_array - self.x)).argmin()
        self.ka = self.ka_array[self.idx]

        if self.button_chain.value == 'monoatomic':
            w= self.w[self.idx]
        elif self.button_chain.value == 'diatomic':
        # Compute if acoustic or optical
            w = self.compute_distance(y=self.y)  

        # Update point position
        self.point.set_data(self.ka, w)
        self.compute_trajectory_1D()

        
    def compute_distance(self, y):
        # Compute vertical distance between point and acoustic/optical branches
        # And return corresponding frequency
        if np.abs((y - self.w_op[self.idx])) < np.abs((y - self.w_ac[self.idx])):
            self.optic = True
            return self.w_op[self.idx]
        else:
            self.optic = False
            return self.w_ac[self.idx]

    def show_slider_M(self,*args):
        if self.button_chain.value == 'monoatomic':
            self.output_ratio.clear_output()
        elif self.button_chain.value == 'diatomic':
            with self.output_ratio:
                display( widgets.HBox([self.slider_M_description,self.slider_M]))