import numpy as np

import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HTML, HBox, Output
from IPython.display import display
from ase import Atoms
from ase.io.trajectory import Trajectory
from sympy import *
from NGLUtilsClass import NGLWidgets
from itertools import product
from scipy.spatial import Voronoi, voronoi_plot_2d

class NGLTrajectory2D(NGLWidgets):
    def __init__(self, trajectory):
        super().__init__(trajectory)
        layout_description=widgets.Layout(width="130px")
        self.slider_amplitude = widgets.FloatSlider(
            value=0.06,
            min=0.01,
            max=0.12,
            step=0.01,
            # description="Amplitude",
            continuous_update=False,
            layout=widgets.Layout(width="200px")
        )
        self.slider_amplitude_description=widgets.Label("Amplitude",layout=layout_description)
        
        self.slider_amplitude.observe(self.recompute_traj, "value")
        self.button_optic=widgets.RadioButtons(
            options=['acoustic', 'optical'],
            value='acoustic',
            # description='Atomic chain type: ',
            disabled=False
        )
        self.button_optic.observe(self.on_band_change_honey,"value")
        eps=1e-4
        kx_array=ky_array = np.linspace(-1.2*np.pi+eps, 1.2*np.pi-2*eps, 25) # Avoid division by zero
        self.kx_array=self.ky_array=np.linspace(-1.2*np.pi,1.2*np.pi,25)

        self.KX, self.KY= np.meshgrid(kx_array,ky_array)
        self.idx_x = 12
        self.idx_y = 12
        self.init_delay=20
        self.nframes=51

        self.ky_honey_array_honey=np.linspace(-1.5*2*np.pi/3,1.5*2*np.pi/3,61)
        self.ky_honey_array=np.linspace(-1.5*4*np.pi/(3*np.sqrt(3)),1.5*4*np.pi/(3*np.sqrt(3)),61)

        self.KX_honey, self.KY_honey= np.meshgrid(self.ky_honey_array_honey,self.ky_honey_array)
        self.idx_x_honey = 15
        self.idx_y_honey = 15
        self.out=widgets.Output()
        # self.initialize_2D_band_plot()
        self.output_plots=widgets.Output()
        self.output_view=widgets.Output()
        self.output_branch=widgets.Output()
        self.button_lattice=widgets.RadioButtons(options=['square','honeycomb'],value='square',description='Lattice')
        self.button_lattice.observe(self.on_change,'value')


    def on_change(self,event):
        with self.output_plots:
            if self.button_lattice.value=='square':
                self.output_plots.clear_output()
                display(HBox([self.fig.canvas,self.fig_.canvas]))
                
            elif self.button_lattice.value=='honeycomb':
                self.output_plots.clear_output()
                display(HBox([self.fig_honey.canvas,self.fig_honey_.canvas]))

        if self.button_lattice.value=='square':
            self.compute_trajectory_2D()
        elif self.button_lattice.value=='honeycomb':
            self.compute_trajectory_2D_honey()
        with self.output_branch:
            if self.button_lattice.value=='square':
                self.output_branch.clear_output()
            elif self.button_lattice.value=='honeycomb':
                self.output_branch.clear_output()
                display(self.button_optic)
            
    def recompute_traj(self):
        if self.button_lattice.value=='square':
            self.compute_trajectory_2D()
        elif self.button_lattice.value=='honeycomb':
            self.compute_trajectory_2D_honey() 

    def addArrows(self, *args):
        self.removeArrows()

        positions = list(self.traj[0].get_positions().flatten())

        n_atoms = int(len(positions) / 3)
        color = n_atoms * [0, 1, 0]
        radius = n_atoms * [0.1]
        self.view._js(
            f"""
        var shape = new NGL.Shape("my_shape")

        var arrowBuffer = new NGL.ArrowBuffer({{position1: new Float32Array({positions}),
        position2: new Float32Array({positions}),
        color: new Float32Array({color}),
        radius: new Float32Array({radius})
        }})

        shape.addBuffer(arrowBuffer)
        globalThis.arrowBuffer = arrowBuffer;
        var shapeComp = this.stage.addComponentFromObject(shape)
        shapeComp.addRepresentation("buffer")
        shapeComp.autoView()
        """
        )
        # Remove observe callable to avoid visual glitch
        if self.handler:
            self.view.unobserve(self.handler.pop(), names=["frame"])

        def on_frame_change(change):
            frame = change["new"]

            positions = self.traj[frame].get_positions()
            positions2 = positions+self.steps[:,:,:,frame].reshape(-1,3)*self.slider_amp_arrow.value
            positions=list(positions.flatten())
            positions2=list(positions2.flatten())
            n_atoms = int(len(positions) / 3)
            radius = n_atoms * [0.1]

            if self.tick_box.value:
                radius = n_atoms * [self.slider_arrow_radius.value]
                self.view._js(
                    f"""
                globalThis.arrowBuffer.setAttributes({{
                position1: new Float32Array({positions}),
                position2: new Float32Array({positions2}),
                radius: new Float32Array({radius})
                }})
                
                this.stage.viewer.requestRender()
                """
                )
            else:
                radius = n_atoms * [0.0]
                self.view._js(
                    f"""
                globalThis.arrowBuffer.setAttributes({{
                position1: new Float32Array({positions}),
                position2: new Float32Array({positions}),
                radius: new Float32Array({radius})
                }})

                this.stage.viewer.requestRender()
                """
                )

        self.view.observe(on_frame_change, names=["frame"])
        self.handler.append(on_frame_change)

    def compute_dispersion(self,*args):

        a=Symbol('a')
        a1=Matrix([a,0])
        a2=Matrix([0,a])
        kx,ky=symbols('k_x k_y')
        ux,uy=symbols('u_x u_y')
        M, C= symbols('M C')
        w, t = symbols('w t')
        atom_positions=Matrix([[1,0],[-1,0],[0,1],[0,-1]])
        atom_positions_x=Matrix([[1,0],[-1,0]])
        atom_positions_y=Matrix([[0,1],[0,-1]])
        RHS1=0*a
        RHS2=0*a
        # d^2ux/dt^2
        for i in range(atom_positions_x.rows):
            position=atom_positions_x.row(i)
            m, n = position 
            k=Matrix([n*kx,m*ky])
            RHS1+=ux*exp(I*(k.T.dot(a1)+k.T.dot(a2)))-ux
        for i in range(atom_positions_y.rows):
            position=atom_positions_y.row(i)
            m, n = position
            k=Matrix([n*kx,m*ky])
            RHS1+=uy*exp(I*(k.T.dot(a1)+k.T.dot(a2)))-uy
        # d^2uy/dt^2
        for i in range(atom_positions_y.rows):
            position=atom_positions_y.row(i)
            m, n = position 
            k=Matrix([n*kx,m*ky])
            RHS2+=uy*exp(I*(k.T.dot(a1)+k.T.dot(a2)))-uy
        for i in range(atom_positions_x.rows):
            position=atom_positions_x.row(i)
            m, n = position
            k=Matrix([n*kx,m*ky])
            RHS2+=ux*exp(I*(k.T.dot(a1)+k.T.dot(a2)))-ux


        RHS1*=-C/M
        RHS2*=-C/M

        RHS1=RHS1.rewrite(cos).simplify().trigsimp()
        RHS2=RHS2.rewrite(cos).simplify().trigsimp()

        matrix = linear_eq_to_matrix([RHS1, RHS2], [ux, uy])[0]
        matrix.simplify()
        eig1, eig2 = matrix.eigenvects() #Eig 1 is zero


        self.A = float(eig2[2][0][0]) # Amplitude ratio is always 1
        self.w2_ = lambdify((kx,ky), eig2[0].subs({C:1,M:1,a:1}))
        self.WW = np.sqrt(self.w2_(self.KX,self.KY))

    def compute_trajectory_2D(self, *args):

        self.kx = self.kx_array[self.idx_x]
        self.ky = self.ky_array[self.idx_y]
        self.w = self.WW[self.idx_x][self.idx_y]

        ax = np.array([1, 0, 0])
        ay = np.array([0, 1, 0])

        amp1=amp2=self.A
        if self.kx==0:
            amp1=0
        if self.ky==0:
            amp2=0
        
        K = np.array([self.kx, self.ky, 0])
        traj = Trajectory("atoms_2d.traj", "w")
        
        self.steps=np.zeros((10,10,3,self.nframes))
        for frame in np.linspace(0, 50, self.nframes):
            atom_positions = []
            if self.w != 0:
                t = 2 * np.pi / self.nframes / self.w * frame
            else:
                t = 0
            for i,j in product(range(0,10),range(0,10)):
                step = np.real(
                            self.slider_amplitude.value
                            * amp1
                            * np.exp(1j * self.w * t)
                            * np.exp(1j * i * np.dot(K, ax)+1j*j*np.dot(K, ay))
                        )* ax + np.real(
                            self.slider_amplitude.value
                            * amp2
                            * np.exp(1j * self.w * t)
                            * np.exp(1j * i * np.dot(K, ax)+1j*j*np.dot(K, ay))
                        ) * ay
                atom_positions_ = (
                        -2.5 * ax
                        + i * ax*0.5
                        -2.5 * ay
                        + j * ay*0.5
                        + step
                    )
                self.steps[i,j,:,int(frame)]+=step

                atom_positions.append(atom_positions_)

            atoms = Atoms(100 * "C", positions=atom_positions)
            traj.write(atoms)

        self.replace_trajectory(
            traj=Trajectory("atoms_2d.traj"), representation="spacefill"
        )
        # self.view.control.zoom(0.25)


    def onclick(self, event):
        x = event.xdata
        y = event.ydata

        # Return idx of closest element in array
        self.idx_x = (np.abs(self.kx_array - x)).argmin()
        self.idx_y = (np.abs(self.ky_array - y)).argmin()

        # Check if point is on plotted path
        if np.any(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(0,np.pi,21),np.linspace(0,np.pi,21)], axis=1)):
            idx=np.where(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(0,np.pi,21),np.linspace(0,np.pi,21)], axis=1))[0][0]
            self.point_.set_data((idx, self.WW[self.idx_x][self.idx_y]))
        elif np.any(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(np.pi,np.pi,21),np.linspace(np.pi,0,21)], axis=1)):
            idx=np.where(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(np.pi,np.pi,21),np.linspace(np.pi,0,21)], axis=1))[0][0]
            idx+=20
            self.point_.set_data((idx, self.WW[self.idx_x][self.idx_y]))
        elif np.any(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(np.pi,0,21),np.linspace(0,0,21)], axis=1)):
            idx=np.where(np.all([self.kx_array[self.idx_x],self.ky_array[self.idx_y]] == np.c_[np.linspace(np.pi,0,21),np.linspace(0,0,21)], axis=1))[0][0]
            idx+=40
            self.point_.set_data((idx, self.WW[self.idx_x][self.idx_y]))
        else: # Point is not on path
            self.point_.set_data([],[])
        # Update point position
        self.point.set_data(self.kx_array[self.idx_x], self.ky_array[self.idx_y])
        self.compute_trajectory_2D()

    def initialize_2D_band_plot(self):
        plt.ioff()
        self.fig, self.ax =plt.subplots(figsize=(4,4))

        self.ax.set_xlim((-1.2*np.pi,1.2*np.pi))
        self.ax.set_ylim((-1.2*np.pi,1.2*np.pi))
        
        self.fig.canvas.toolbar_visible = False
        self.fig.canvas.header_visible = False
        self.fig.canvas.footer_visible = False

        # Diagonals
        self.ax.plot([0, 1], [0, 1], transform=self.ax.transAxes, linestyle='--',c='black',linewidth=0.5)
        self.ax.plot([0 ,1], [1, 0], transform=self.ax.transAxes, linestyle='--',c='black',linewidth=0.5)
        # Paths
        self.ax.plot([0,np.pi],[0,np.pi],'--',c='blue',linewidth=2.5)
        self.ax.plot([np.pi,np.pi],[np.pi,0],'--',c='blueviolet',linewidth=2.5)
        self.ax.plot([np.pi,0],[0,0],'--',c='violet',linewidth=2.5)
        # First Brillouin zone
        self.ax.plot([-np.pi,np.pi],[-np.pi,-np.pi],'k',linewidth=2)
        self.ax.plot([-np.pi,np.pi],[np.pi,np.pi],'k',linewidth=2)
        self.ax.plot([np.pi,np.pi],[-np.pi,np.pi],'k',linewidth=2)
        self.ax.plot([-np.pi,-np.pi],[-np.pi,np.pi],'k',linewidth=2)

        self.ax.axvline(0,linestyle='--',c='black',linewidth=0.5)
        self.ax.axhline(0,linestyle='--',c='black',linewidth=0.5)
        self.ax.text(-0.4,-0.5,'$\mathbf{\Gamma}$',fontsize=16)
        self.ax.plot(0,0,'r.')
        self.ax.text(np.pi-0.5,-0.5,'$\mathbf{X}$',fontsize=16)
        self.ax.plot(np.pi,0,'r.')
        self.ax.text(np.pi-0.8,np.pi-0.5,'$\mathbf{M}$',fontsize=16)
        self.ax.plot(np.pi,np.pi,'r.')
        self.ax.set_xlabel("k$_x$")
        self.ax.set_ylabel("k$_y$")
        self.ax.set_xticks(np.linspace(-np.pi,np.pi,5))
        self.ax.set_xticklabels(['$-\pi/a$','','0','','$\pi/a$'])
        self.ax.set_yticks(np.linspace(-np.pi,np.pi,5))
        self.ax.set_yticklabels(['$-\pi/a$','','0','','$\pi/a$'])

        self.point,=self.ax.plot([0],[0],'.',c='crimson',markersize=10)

        

        self.fig.canvas.mpl_connect('button_press_event', self.onclick);
        plt.ion();

    def onclick_(self,event):
        x = event.xdata

        if x<20:
            idx=round(x)
            kx=np.linspace(0,np.pi,21)
            ky=np.linspace(0,np.pi,21)
            y=np.sqrt(self.w2_(kx[idx],ky[idx]))
        elif 20<=x<40:
            idx=round(x)-20
            kx=np.linspace(np.pi,np.pi,21)
            ky=np.linspace(np.pi,0,21)
            y=np.sqrt(self.w2_(kx[idx],ky[idx]))
        elif x>=40:
            idx=round(x)-40
            kx=np.linspace(np.pi,0,21)
            ky=np.linspace(0,0,21)
            y=np.sqrt(self.w2_(kx[idx],ky[idx]))

        
        self.idx_x = (np.abs(self.kx_array - kx[idx])).argmin()
        self.idx_y = (np.abs(self.ky_array - ky[idx])).argmin()

        self.point_.set_data((round(x),y))
        self.point.set_data((kx[idx],ky[idx]))

        self.compute_trajectory_2D()
        
    def initialize_paths_bands(self):
        plt.ioff()
        w_GM=np.sqrt(self.w2_(np.linspace(0,np.pi,21),np.linspace(0,np.pi,21)))
        w_MX=np.sqrt(self.w2_(np.linspace(np.pi,np.pi,21),np.linspace(np.pi,0,21)))
        w_XG=np.sqrt(self.w2_(np.linspace(np.pi,0,21),np.linspace(0,0,21)))

        self.fig_,self.ax_=plt.subplots(figsize=(4,4))
        self.fig_.canvas.toolbar_visible = False
        self.fig_.canvas.header_visible = False
        self.fig_.canvas.footer_visible = False

        self.ax_.plot(np.linspace(0,20,21),w_GM,c='blue')
        self.ax_.plot(np.linspace(40,60,21),w_XG,c='violet')
        self.ax_.plot(np.linspace(20,40,21),w_MX,c='blueviolet')
        self.ax_.plot([20,20],[0,3],'k--')
        self.ax_.plot([40,40],[0,3],'k--')

        self.point_,=self.ax_.plot([],[],'r.',markersize=10)
        self.ax_.set_xticks([0,20,40,60])
        self.ax_.set_xticklabels(['$\mathbf{\Gamma}$','$\mathbf{M}$','$\mathbf{X}$','$\mathbf{\Gamma}$'])
        self.ax_.set_ylim(0,w_GM[-1]+1e-2);

        self.fig_.canvas.mpl_connect('button_press_event', self.onclick_);
        plt.ion();



    def compute_dispersion_honey(self,*args):

        a=Symbol('a') # a is next neighboors lattice param
        a1=Matrix([3/2*a,sqrt(3)/2*a])
        a2=Matrix([3/2*a,-sqrt(3)/2*a])

        ux,uy,vx,vy= symbols('u_x u_y v_x v_y')
        kx,ky=symbols('kx ky')
        M, C= symbols('M C')
        k=Matrix([kx,ky])
        RHS1x=0*a
        RHS1y=0*a
        RHS2x=0*a
        RHS2y=0*a
        atom_positions1=Matrix([[0, 0],[0,-1],[-1,0]]) # In a1 a2 basis
        atom_positions2=Matrix([[0,0],[1,0],[0,1]]) # In a1 a2 basis

        for i in range(atom_positions1.rows):
            position=atom_positions1.row(i)
            m, n = position 
            RHS1x+=vx*exp(I*(n*k.T.dot(a1)+m*k.T.dot(a2)))-ux

        for i in range(atom_positions1.rows):
            position=atom_positions1.row(i)
            m, n = position 
            RHS1y+=vy*exp(I*(n*k.T.dot(a1)+m*k.T.dot(a2)))-uy

        for i in range(atom_positions2.rows):
            position=atom_positions2.row(i)
            m, n = position 
            RHS2x+=ux*exp(I*(n*k.T.dot(a1)+m*k.T.dot(a2)))-vx

        for i in range(atom_positions2.rows):
            position=atom_positions2.row(i)
            m, n = position 
            RHS2y+=uy*exp(I*(n*k.T.dot(a1)+m*k.T.dot(a2)))-vy

        RHS1x*=-C/M
        RHS1y*=-C/M
        RHS2x*=-C/M
        RHS2y*=-C/M

        matrix=linear_eq_to_matrix([RHS1x,RHS1y,RHS2x,RHS2y], [ux,uy,vx,vy])[0]
        self.numpy_matrix=lambdify((kx,ky),matrix.subs({'M':1,'C':1,'a':1}))


        self.w_honey_opt=np.zeros((61,61),dtype='complex128')
        self.w_honey_acc=np.zeros((61,61),dtype='complex128')
        self.A_opt=np.zeros((61,61,4),dtype='complex128')
        self.A_acc=np.zeros((61,61,4),dtype='complex128')
        for i in range(61):
            for j in range(61):
                matrice=self.numpy_matrix(self.ky_honey_array_honey[i],self.ky_honey_array[j])
                w2,v=np.linalg.eig(matrice)
                w_1,v_1=np.sqrt(w2[2]),v[:,2]
                w_2,v_2=np.sqrt(w2[3]),v[:,3]
                if w_1>w_2:
                    self.w_honey_opt[i][j]+=w_1
                    self.A_opt[i][j]+=v_1
                    self.w_honey_acc[i][j]+=w_2
                    self.A_acc[i][j]+=v_2
                else:
                    self.w_honey_opt[i][j]+=w_2
                    self.A_opt[i][j]+=v_2
                    self.w_honey_acc[i][j]+=w_1
                    self.A_acc[i][j]+=v_1



        # self.A = float(eig2[2][0][0]) # Amplitude ratio is always 1
        # self.w_honey2_ = lambdify((kx,ky), eig2[0].subs({C:1,M:1,a:1}))
        # self.WW = np.sqrt(self.w_honey2_(self.KX_honey,self.KY_honey))

    def compute_trajectory_2D_honey(self, *args):

        self.kx_honey = self.ky_honey_array_honey[self.idx_x_honey]
        self.ky_honey = self.ky_honey_array[self.idx_y_honey]

        if self.button_optic.value=='optical':
            self.w_honey = self.w_honey_opt[self.idx_x_honey][self.idx_y_honey]
            ux,uy,vx,vy=self.A_opt[self.idx_x_honey][self.idx_y_honey]
        elif self.button_optic.value=='acoustic':
            self.w_honey = self.w_honey_acc[self.idx_x_honey][self.idx_y_honey]
            ux,uy,vx,vy=self.A_acc[self.idx_x_honey][self.idx_y_honey]

        a=1
        ax=np.array([3/2*a,np.sqrt(3)/2*a,0])
        ay=np.array([3/2*a,-np.sqrt(3)/2*a,0])

        u=np.array([ux,uy,0])
        v=np.array([vx,vy,0])

        k=np.array([self.kx_honey,self.ky_honey,0])

        traj = Trajectory("atoms_2d_honeycomb.traj", "w")
        self.steps=np.zeros((10,5,3,self.nframes))
        self.steps_honey_1=np.zeros((5,5,3,self.nframes))
        self.steps_honey_2=np.zeros((5,5,3,self.nframes))
        for frame in np.linspace(0, 50, self.nframes):
            atom_positions=[]
            t = 2 * np.pi / self.nframes / self.w_honey * frame
            for i,j in product(range(-2,3),range(-2,3)):
                step_1=np.real(u*5*self.slider_amplitude.value*np.exp(1j*(i*k@ax+j*k@ay-self.w_honey*t)))
                atom_positions_1=(
                                + i * ax
                                + j * ay
                                +np.array([a/2 ,0,0])
                                +step_1)
                self.steps_honey_1[i+2,j+2,:,int(frame)]+=step_1
                self.steps[i+2,j+2,:,int(frame)]+=step_1
                atom_positions.append(atom_positions_1)
            for i,j in product(range(-2,3),range(-2,3)):
                step_2=np.real(v*5*self.slider_amplitude.value*np.exp(1j*(i*k@ax+j*k@ay-self.w_honey*t)))    
                atom_positions_2=(
                                + i * ax
                                + j * ay
                                +np.array([-a/2,0,0])
                                +step_2)
                self.steps_honey_2[i+2,j+2,:,int(frame)]+=step_2
                self.steps[i+7,j+2,:,int(frame)]+=step_2
                atom_positions.append(atom_positions_2)
                
            atoms = Atoms(len(atom_positions) * "C", positions=atom_positions)
            traj.write(atoms)

        self.replace_trajectory(
            traj=Trajectory("atoms_2d_honeycomb.traj"), representation="spacefill"
        )
        # self.view.control.zoom(0.25)


    def onclick_honey(self, event):
        self.x_honey = event.xdata
        self.y_honey = event.ydata
        
        # Return idx of closest element in array
        self.idx_x_honey = (np.abs(self.ky_honey_array_honey - self.x_honey)).argmin()
        self.idx_y_honey = (np.abs(self.ky_honey_array - self.y_honey)).argmin()

        # Check if point is on plotted path
        if np.any(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]),np.c_[np.linspace(0,self.point_honeys_hexagon[3,0],11),np.linspace(0,self.point_honeys_hexagon[3,1],11)]), axis=1)):
            idx=np.where(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]),np.c_[np.linspace(0,self.point_honeys_hexagon[3,0],11),np.linspace(0,self.point_honeys_hexagon[3,1],11)]), axis=1))[0][0]
            if self.button_optic.value=='acoustic':
                self.point_honey_.set_data((idx, self.w_honey_GK_acc[idx]))
            else:
                self.point_honey_.set_data((idx, self.w_honey_GK_opt[idx]))

        elif np.any(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]), np.c_[np.linspace(self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,0],11),np.linspace(self.point_honeys_hexagon[3,1],0,11)]), axis=1)):
            idx=np.where(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]), np.c_[np.linspace(self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,0],11),np.linspace(self.point_honeys_hexagon[3,1],0,11)]), axis=1))[0][0]
            # idx+=10
            if self.button_optic.value=='acoustic':
                self.point_honey_.set_data((idx+10, self.w_honey_KM_acc[idx]))
            else:
                self.point_honey_.set_data((idx+10, self.w_honey_KM_opt[idx]))
        elif np.any(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]) ,np.c_[np.linspace(self.point_honeys_hexagon[3,0],0,21),np.linspace(0,0,21)]), axis=1)):
            idx=np.where(np.all(np.isclose(np.array([self.ky_honey_array_honey[self.idx_x_honey],self.ky_honey_array[self.idx_y_honey]]) ,np.c_[np.linspace(self.point_honeys_hexagon[3,0],0,21),np.linspace(0,0,21)]), axis=1))[0][0]
            # idx+=20
            if self.button_optic.value=='acoustic':
                self.point_honey_.set_data((idx+20, self.w_honey_MG_acc[idx]))
            else:
                self.point_honey_.set_data((idx+20, self.w_honey_MG_opt[idx]))
        else: # Point is not on path
            self.point_honey_.set_data([],[])
        # Update point position
        self.point_honey.set_data(self.ky_honey_array_honey[self.idx_x_honey], self.ky_honey_array[self.idx_y_honey])
        self.compute_trajectory_2D_honey()

    def initialize_2D_band_plot_honey(self):
        plt.ioff()
        bx = 2*np.pi*np.array([1/3,1/np.sqrt(3)])
        by = 2*np.pi*np.array([1/3,-1/np.sqrt(3)])

        points = np.array([[0, 0], bx, -bx, by,-by, bx+by, -bx-by ])
        vor = Voronoi(points)

        # Taken from vor.vertices and reorganized
        self.point_honeys_hexagon=np.array([[-2*np.pi/3 , -2*np.pi/(3*np.sqrt(3))],
            [ 0.        , -4*np.pi/(3*np.sqrt(3))],
            [ 2*np.pi/3 , -2*np.pi/(3*np.sqrt(3))],
            [ 2*np.pi/3 ,  2*np.pi/(3*np.sqrt(3))],
            [ 0.        ,  4*np.pi/(3*np.sqrt(3))],
            [-2*np.pi/3 ,  2*np.pi/(3*np.sqrt(3))],
            ])

        self.fig_honey = voronoi_plot_2d(vor,show_points=False,show_vertices=False)
        px = 1 / plt.rcParams["figure.dpi"]
        self.fig_honey.canvas.toolbar_visible = False
        self.fig_honey.canvas.header_visible = False
        self.fig_honey.canvas.footer_visible = False

        self.fig_honey.set_figheight(400*px)
        self.fig_honey.set_figwidth(400*px)
        self.ax_honey,=self.fig_honey.axes
        self.ax_honey.set_xlim((-3,3))
        self.ax_honey.set_ylim((-3,3));

        self.ax_honey.plot([self.point_honeys_hexagon[0,0],self.point_honeys_hexagon[3,0]],[self.point_honeys_hexagon[0,1],self.point_honeys_hexagon[3,1]],'k--',linewidth=1)
        self.ax_honey.plot([self.point_honeys_hexagon[1,0],self.point_honeys_hexagon[4,0]],[self.point_honeys_hexagon[1,1],self.point_honeys_hexagon[4,1]],'k--',linewidth=1)
        self.ax_honey.plot([self.point_honeys_hexagon[2,0],self.point_honeys_hexagon[5,0]],[self.point_honeys_hexagon[2,1],self.point_honeys_hexagon[5,1]],'k--',linewidth=1)

        self.ax_honey.plot([0,self.point_honeys_hexagon[3,0]],[0,self.point_honeys_hexagon[3,1]],'--',c='blue',linewidth=2.5)
        self.ax_honey.plot([self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,0]],[self.point_honeys_hexagon[3,1],0],'--',c='blueviolet',linewidth=2.5)
        self.ax_honey.plot([self.point_honeys_hexagon[3,0],0],[0,0],'--',c='violet',linewidth=2.5)

        self.point_honey,=self.ax_honey.plot([0],[0],'.',c='crimson',markersize=10)

        self.ax_honey.text(-0.2,-0.5,'$\mathbf{\Gamma}$',fontsize=16)
        self.ax_honey.plot(0,0,'r.')
        self.ax_honey.text(self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,1]+0.2,'$\mathbf{K}$',fontsize=16)
        self.ax_honey.plot(self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,1],'r.')
        self.ax_honey.text(self.point_honeys_hexagon[3,0]+0.02,0-0.4,'$\mathbf{M}$',fontsize=16)
        self.ax_honey.plot(self.point_honeys_hexagon[3,0],0,'r.')

        self.fig_honey.canvas.mpl_connect('button_press_event', self.onclick_honey);
        plt.ion();

    def onclick_honey_(self,event):
        self.x_honey_ = event.xdata

        if self.x_honey_<10:
            idx=round(self.x_honey_)
            kx=np.linspace(0,self.point_honeys_hexagon[3,0],11)
            ky=np.linspace(0,self.point_honeys_hexagon[3,1],11)
            if self.button_optic.value=='acoustic':
                y=self.w_honey_GK_acc[idx]
            else:
                y=self.w_honey_GK_opt[idx]
        elif 10<=self.x_honey_<20:
            idx=round(self.x_honey_)-10
            kx=np.linspace(self.point_honeys_hexagon[3,0],self.point_honeys_hexagon[3,0],11)
            ky=np.linspace(self.point_honeys_hexagon[3,1],0,11)
            if self.button_optic.value=='acoustic':
                y=self.w_honey_KM_acc[idx]
            else:
                y=self.w_honey_KM_opt[idx]
        elif self.x_honey_>=20:
            idx=round(self.x_honey_)-20
            kx=np.linspace(self.point_honeys_hexagon[3,0],0,21)
            ky=np.linspace(0,0,21)
            if self.button_optic.value=='acoustic':
                y=self.w_honey_MG_acc[idx]
            else:
                y=self.w_honey_MG_opt[idx]

        
        self.idx_x_honey = (np.abs(self.ky_honey_array_honey - kx[idx])).argmin()
        self.idx_y_honey = (np.abs(self.ky_honey_array - ky[idx])).argmin()

        self.point_honey_.set_data((round(self.x_honey_),y))
        self.point_honey.set_data((kx[idx],ky[idx]))

        self.compute_trajectory_2D_honey()
        
    def initialize_paths_bands_honey(self):
        plt.ioff()
        
        self.fig_honey_,self.ax_honey_=plt.subplots(figsize=(4,4))
        
        self.fig_honey_.canvas.toolbar_visible = False
        self.fig_honey_.canvas.header_visible = False
        self.fig_honey_.canvas.footer_visible = False

        self.w_honey_GK_acc=self.w_honey_acc[np.arange(30,self.nframes,2),np.arange(30,41)]
        self.w_honey_KM_acc=self.w_honey_acc[50,40:29:-1]
        self.w_honey_MG_acc=self.w_honey_acc[50:29:-1,30]

        self.w_honey_GK_opt=self.w_honey_opt[np.arange(30,self.nframes,2),np.arange(30,41)]
        self.w_honey_KM_opt=self.w_honey_opt[50,40:29:-1]
        self.w_honey_MG_opt=self.w_honey_opt[50:29:-1,30]

        self.line_GK_acc,=self.ax_honey_.plot(np.linspace(0,10,11),self.w_honey_GK_acc,'blue') # GK
        self.line_KM_acc,=self.ax_honey_.plot(np.linspace(10,20,11),self.w_honey_KM_acc,'blueviolet') # KM
        self.line_MG_acc,=self.ax_honey_.plot(np.linspace(20,40,21),self.w_honey_MG_acc,'violet') # MG
        self.line_GK_opt,=self.ax_honey_.plot(np.linspace(0,10,11),self.w_honey_GK_opt,'blue') # GK
        self.line_KM_opt,=self.ax_honey_.plot(np.linspace(10,20,11),self.w_honey_KM_opt,'blueviolet') # KM
        self.line_MG_opt,=self.ax_honey_.plot(np.linspace(20,40,21),self.w_honey_MG_opt,'violet') # MG

        self.line_GK_opt.set_alpha(0.2)
        self.line_KM_opt.set_alpha(0.2)
        self.line_MG_opt.set_alpha(0.2)
        # self.ax_honey_.plot(np.linspace(0,20,21),w_GM,c='blue')
        # self.ax_honey_.plot(np.linspace(40,60,21),w_XG,c='violet')
        # self.ax_honey_.plot(np.linspace(20,40,21),w_MX,c='blueviolet')
        self.ax_honey_.plot([10,10],[0,3],'k--')
        self.ax_honey_.plot([20,20],[0,3],'k--')

        self.point_honey_,=self.ax_honey_.plot([],[],'r.',markersize=10)
        self.ax_honey_.set_xticks([0,10,20,40])
        self.ax_honey_.set_xticklabels(['$\mathbf{\Gamma}$','$\mathbf{K}$','$\mathbf{M}$','$\mathbf{\Gamma}$'])
        self.ax_honey_.set_ylim(0,self.w_honey_GK_opt[0]+1e-2);
        self.ax_honey_.set_xlim(0,40)
        self.fig_honey_.canvas.mpl_connect('button_press_event', self.onclick_honey_);
        plt.ion();

    def on_band_change_honey(self,*args):
        if self.button_optic.value=='acoustic':
            self.line_GK_acc.set_alpha(1)
            self.line_KM_acc.set_alpha(1)
            self.line_MG_acc.set_alpha(1)
            self.line_GK_opt.set_alpha(0.2)
            self.line_KM_opt.set_alpha(0.2)
            self.line_MG_opt.set_alpha(0.2)
            self.point_honey.set_data([0],[0])
            self.point_honey_.set_data([0],self.w_honey_GK_acc[0])
        else:
            self.line_GK_acc.set_alpha(0.2)
            self.line_KM_acc.set_alpha(0.2)
            self.line_MG_acc.set_alpha(0.2)
            self.line_GK_opt.set_alpha(1)
            self.line_KM_opt.set_alpha(1)
            self.line_MG_opt.set_alpha(1)
            self.point_honey.set_data([0],0)
            self.point_honey_.set_data([0],self.w_honey_GK_opt[0])
        

        self.idx_x_honey=30
        self.idx_y_honey=30

        self.compute_trajectory_2D_honey()