from xml.parsers.expat import model
import nglview as nv
import threading
import moviepy.editor as mpy
import time
import tempfile
import numpy as np
import functools
import os

import ipywidgets as widgets
from ipywidgets import HTML, Label, HBox, VBox,IntSlider,HTMLMath,Output, Layout

from ipyevents import Event
import base64

from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS
from ase.vibrations import Vibrations
from ase.io.trajectory import Trajectory
from ase.units import kB

from NGLUtilsClass import NGLWidgets


class NGLMolecule(NGLWidgets):
    def __init__(self, trajectory) -> None:
        super().__init__(trajectory)

        layout = widgets.Layout(width="50px")
        layout_molecule=Layout(width='200px')
        style = {"description_width": "initial"}

        self.output_summary = widgets.Output()

        # Molecules
        self.slider_mode_description=widgets.HTMLMath(r"Vibrational mode",layout=self.layout_description)
        self.slider_mode = widgets.SelectionSlider(
            options=['1/1'],
            value='1/1',
            # description="Vibrational mode",
            style={"description_width": "initial"},
            continuous_update=False,
            layout=layout_molecule,
            readout=True
        )
        
        self.slider_amplitude_description=widgets.HTMLMath(r"Temperature [K]",layout=self.layout_description)
        self.slider_amplitude = widgets.FloatSlider(
            value=300,
            min=10,
            max=1000,
            step=10,
            # description="Temperature [K]",
            continuous_update=False,
            style=style,
            layout=layout_molecule
        )
        
        self.dropdown_molecule_description=widgets.HTMLMath(r"Molecule",layout=self.layout_description)
        self.dropdown_molecule = widgets.Dropdown(
            options=["O\u2082", "N\u2082", "OH", "H\u2082O", "CO\u2082", "CH\u2084", "NH\u2083","C\u2086H\u2086"],
            value="O\u2082",
            # description="Molecule:",
            disabled=False,
            # style={"font_weight": "bold"},
            layout=layout_molecule
        )
        
        self.dropdown_molecule.observe(self.change_molecule, "value")
        self.slider_mode.observe(self.modify_molecule, "value")
        self.slider_amplitude.observe(self.modify_molecule, "value")

        self.molecule = widgets.HBox(
            [
                widgets.VBox(
                    [HBox([self.dropdown_molecule_description,self.dropdown_molecule]), HBox([self.slider_amplitude_description,self.slider_amplitude]), HBox([self.slider_mode_description,self.slider_mode])]
                ),
                self.output_summary,
            ]
        )

        for widget in [self.slider_mode, self.dropdown_molecule, self.slider_amplitude]:
            self.widgetList.append(widget)
        self.representation='ball+stick'
        
        self.slider_aspect_ratio=widgets.FloatSlider(value=2,min=1,max=3,step=0.1,layout=Layout(width='200px'))
        self.slider_aspect_ratio_description=HTMLMath(r"Aspect ratio",layout=self.layout_description)
        self.slider_aspect_ratio.observe(self.modifiy_representation,"value")

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

        scaling_factor=np.max(np.linalg.norm(self.steps[:,:,:,:],axis=2))
        def on_frame_change(change):
            frame = change["new"]
            
            positions = self.traj[frame].get_positions()
            positions2 = positions+self.steps[:,:,:,frame].reshape(-1,3)/scaling_factor*self.slider_amp_arrow.value
            positions=list(positions.flatten())
            positions2=list(positions2.flatten())

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

    def modify_molecule(self, *args):
        """
        Example slider:
            slider_amplitude=FloatSlider(value=300,min=10,max=1000,step=10,description='Temperature [K]',continuous_update=False)
            slider_amplitude.observe(functools.partial(set_amplitude,view,slider))
        """

        time.sleep(0.2)
        
        molecule_name = ""
        for x in self.dropdown_molecule.value:
            if ord(x) > 128:
                # Retrieve number from unicode
                molecule_name += chr(ord(x) - 8320 + 48)
            else:
                molecule_name += x

        atoms = molecule(molecule_name, calculator=EMT())
        vibname = os.path.join(self.folder, molecule_name)
        vib = Vibrations(atoms, name=vibname)
        # TODO : write in temporary file
        
        mode=int(self.slider_mode.value[0])
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode-1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                molecule_name + "." + str(self.idx[mode-1]) + ".traj",
            )
        )

        self.steps=np.zeros((len(traj[0].positions),1,3,60))
        
        for frame in range(len(traj)):
            step=traj[frame].positions-traj[0].positions
            self.steps[:,0,:,frame]=step

        self.replace_trajectory(traj=traj)

        self.print_summary()

    def change_molecule(self, *args):
        time.sleep(0.2)  # Time to get the dropdown value to update
        molecule_name = ""
        for x in self.dropdown_molecule.value:
            if ord(x) > 128:
                # Retrieve number from unicode
                molecule_name += chr(ord(x) - 8320 + 48)
            else:
                molecule_name += x

        atoms = molecule(molecule_name, calculator=EMT())

        # Relax and get vibrational properties
        opt = BFGS(atoms, logfile=None)
        opt.run(fmax=0.001)

        vibname = os.path.join(self.folder, molecule_name)
        vib = Vibrations(atoms, name=vibname)
        vib.run()

        # Extract rotational motions
        ndofs = 3 * len(atoms)
        is_not_linear = int(not (len(atoms) == 2 or atoms.get_angle(0, 1, 2) == 0))
        nrotations = ndofs - 5 - is_not_linear
        energies = np.absolute(vib.get_energies())
        frequencies = np.absolute(vib.get_frequencies())

        # Get the nrotations-largest energies, to eliminate translation and rotation energies
        self.idx = np.argpartition(energies, -nrotations)[-nrotations:]
        self.energies = energies[self.idx]
        self.frequencies = frequencies[self.idx]
        options=[]
        max_val=len(self.idx)
        for i in range(1,max_val+1):
            options.append(str(i)+"/"+str(max_val))
        # self.slider_mode.max = len(self.idx) - 1
        self.slider_mode.options=options
        
        mode=int(self.slider_mode.value[0])

        # TODO : write in temporary file
        T = self.slider_amplitude.value
        vib.write_mode(n=self.idx[mode-1], kT=kB * T, nimages=60)

        traj = Trajectory(
            os.path.join(
                self.folder,
                molecule_name + "." + str(self.idx[mode-1]) + ".traj",
            )
        )
        self.steps=np.zeros((len(traj[0].positions),1,3,60))

        for frame in range(len(traj)):
            step=traj[frame].positions-traj[0].positions
            self.steps[:,0,:,frame]=step

        self.replace_trajectory(traj=traj)

        self.print_summary()

    def print_summary(self, *args):
        mode=int(self.slider_mode.value[0])
        with self.output_summary:
            self.output_summary.clear_output()
            titles=HBox([
            HTMLMath(value=1*' ',layout=Layout(width="100px")), #Spacer
            HTMLMath(value='Energy [meV]',layout=self.layout_description),  
            HTMLMath(value=r'Frequency [cm$^{-1}$]',layout=self.layout_description),
            HTMLMath(value='Raman active ',layout=self.layout_description),
            HTMLMath(value='IR active',layout=self.layout_description)
            ])
            values=HBox([
            HTMLMath(value=1*' ',layout=Layout(width="100px")), #Spacer
            HTMLMath(value=f"{self.energies[mode-1]:.3f}",layout=self.layout_description),  
            HTMLMath(value=f"{self.frequencies[mode-1]:.0f}",layout=self.layout_description),
            HTMLMath(value='Unknown ',layout=self.layout_description),
            HTMLMath(value='Unknown',layout=self.layout_description)
            ])
            display(titles,values)
            # print(f"{'Energy [meV]':>15}{'Frequency [cm$^-{1}$]':>25}")
            # print(
            #     f"{self.energies[mode-1]:>15.3f}{self.frequencies[mode-1]:>25.0f}"
            # )

    def change_mode(self, *args):   
        mode=int(self.slider_mode.value[0])
        molecule_name = self.dropdown_molecule.value
        traj = Trajectory(
            os.path.join(
                self.folder,
                molecule_name + "." + str(self.idx[mode-1]) + ".traj",
            )
        )
        self.replace_trajectory(traj=traj)
