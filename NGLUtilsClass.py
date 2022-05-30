import nglview as nv
import threading
import moviepy.editor as mpy
import time
import tempfile
import numpy as np
import functools
import os
import shutil

import ipywidgets as widgets
from ipywidgets import HTML, Label, HBox, VBox,IntSlider,HTMLMath,Output, Layout

from ipyevents import Event
import base64

from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
from IPython.display import Image

import json
class NGLWidgets:

    def __init__(self, trajectory) -> None:
        
        self.tmp_dir=tempfile.TemporaryDirectory(prefix='user_', dir='.')
        self.traj = trajectory

        self.view = nv.show_asetraj(self.traj)
        self.flag = True
        self.name = []
        self.handler = []
        
        self.idx = [5]
        self.zoom = 15
        self.layout_description=widgets.Layout(width="130px")
        self.layout=Layout(width='200px')
        # Camera

        layout_camera = widgets.Layout(width="50px")
        style = {"description_width": "initial"}
        self.button_x = widgets.Button(description="x", layout=layout_camera)
        self.button_y = widgets.Button(description="y", layout=layout_camera)
        self.button_z = widgets.Button(description="z", layout=layout_camera)
        self.camera_upload = widgets.FileUpload(
            accept=".txt",  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'
            multiple=False,  # True to accept multiple files upload else False
            description="Import camera position",
        )
        
        # Outputs
        self.output_text = widgets.Output()
        self.output_movie = widgets.Output()
        self.output_camera = widgets.Output()
        self.output_gif= widgets.Output()
        self.gif_link=widgets.HTML()
        self.output_download_button=Output()

        self.output_camera_position=Output()
        self.output_camera_position_error=Output()
        # Camera
        self.button_x.on_click(functools.partial(self.set_camera, direction="x"))
        self.button_y.on_click(functools.partial(self.set_camera, direction="y"))
        self.button_z.on_click(functools.partial(self.set_camera, direction="z"))
        
        # Animation
        self.dropdown_resolution_description=HTMLMath(r"Resolution",layout=self.layout_description)
        self.dropdown_resolution = widgets.Dropdown(
            options=["480p", "720p", "1080p", "1440p", "2K", "500x500p", "1000x1000p"],
            value="1080p",
            # description="Resolution",
            disabled=False,
            layout=self.layout
            # layout=widgets.Layout(width="300px")
        )
        self.slider_speed_description=HTMLMath(r"Animation speed",layout=self.layout_description)
        self.slider_speed = widgets.FloatSlider(
            value=1,
            min=0.1,
            max=2,
            step=0.1,
            # description="Animation speed",
            continuous_update=False,
            layout=self.layout,
            # style=style,
        )

        self.slider_speed.observe(self.set_speed, "value")
        # Movie
        self.button_movie = widgets.Button(description="Render GIF", style=style,layout=self.layout_description)
        self.button_download = HTML(
            '<html><body><button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download File</button> </body></html>'
        )
        # self.tick_gif=widgets.Checkbox(value=False, description="Show preview")

        # self.event_hide = Event(source=self.button_download, watched_events=["click"])
        # self.event_change = Event(
        #     source=self.button_download, watched_events=["mouseenter"]
        # )
        self.event_change = Event(
            source=self.button_download, watched_events=["click"]
        )
        # self.event_hide.on_dom_event(self.hide_button)
        # self.event_change.on_dom_event(self.change_movie)
        self.event_change.on_dom_event(self.remove_movie)

        self.button_movie.on_click(self.make_movie)
        self.button_download.layout.visibility = "hidden"
        # self.tick_gif.observe(self.show_gif_preview,"value")
        # Arrows
        self.slider_amp_arrow_description=HTMLMath(r"Arrow amplitude",layout=self.layout_description)
        self.slider_amp_arrow = widgets.FloatSlider(
            value=2.0,
            min=0.1,
            max=5.01,
            step=0.1,
            # description=f'{"Arrow amplitude":<20}',
            continuous_update=False,
            # style={"description_width": "100px"},
            layout=self.layout,
        )
        self.tick_box_description=HTMLMath(r"Show arrows",layout=self.layout_description)
        self.tick_box = widgets.Checkbox(
            value=False,
            #  description="Show arrows",
            #  style={"description_width": "100px"}

        )

        self.slider_arrow_radius_description=HTMLMath(r"Arrow radius",layout=self.layout_description)
        self.slider_arrow_radius = widgets.FloatSlider(
            value=0.2,
            min=0.01,
            max=0.3,
            step=0.01,
            # description=f'{"Arrow radius":<20}',
            continuous_update=False,
            # style={"description_width": "100px"},
            layout=self.layout,
        )

        self.camera = widgets.VBox(
            [
                widgets.Label(value="$\Large{\\text{Camera view}}$"),
                widgets.HBox([self.button_x, self.button_y, self.button_z]),
                self.slider_speed,
            ]
        )
        self.arrow = widgets.VBox(
            [HBox([self.slider_amp_arrow_description,self.slider_amp_arrow]), HBox([self.slider_arrow_radius_description,self.slider_arrow_radius]), HBox([self.tick_box_description,self.tick_box])]
        )

        self.movie = VBox(
            [
                HBox([self.button_movie, self.button_download]),
                HBox([self.dropdown_resolution_description,self.dropdown_resolution]),
                HBox([self.slider_speed_description,self.slider_speed]),
                self.output_movie,
                # self.tick_gif
            ]
        )

        self.widgetList = [
            self.button_x,
            self.button_y,
            self.button_z,
            self.button_movie,
            self.slider_speed,
            self.slider_amp_arrow,
            self.slider_arrow_radius,
            self.tick_box,
            self.dropdown_resolution,
        ]
        if self.tick_box.value:
             self.addArrows()
        self.init_delay=25
        self.tmpFileName_movie=None
        # self.tmpdir=tempfile.TemporaryDirectory(prefix='user_',dir='.')
        self.tmpdir='.'
        self.representation='spacefill'
        self.slider_atom_radius_description=HTMLMath(r"Atom radius",layout=self.layout_description)
        self.slider_atom_radius = widgets.FloatSlider(
            value=0.2,
            min=0.01,
            max=0.25,
            step=0.01,
            continuous_update=False,
            layout=self.layout,
        )
        self.slider_atom_radius.observe(self.modifiy_representation,"value")
        self.view.observe(self.on_orientation_change, names=['_camera_orientation'])
        self.text_orientation=widgets.Textarea(value='Paste camera orientation here')
        self.camera_orientation_description=HTMLMath(r"Camera orientation :",layout=self.layout_description)
        self.text_orientation.observe(self.change_camera_position,names=['value'])

    def on_orientation_change(self,*args):
        with self.output_camera_position:
            self.output_camera_position.clear_output()
            position=[round(x,1) for x in self.view._camera_orientation]
            print(position)

    def change_camera_position(self,*args):
        orientation=json.loads(self.text_orientation.value)
        with self.output_camera_position_error:
            self.output_camera_position_error.clear_output()
            if type(orientation) is not list or len(orientation)!=16:
                print("Orientation must be a length 16 list")
            else:        
                self.view._set_camera_orientation(orientation)


    def set_camera(self, *args, direction="x"):
        # See here for rotation matrix https://www.brainvoyager.com/bv/doc/UsersGuide/CoordsAndTransforms/SpatialTransformationMatrices.html

        """
        Button example:
        button_x = widgets.Button(description="x")
        button_x.on_click(functools.partial(set_camera, view ,direction="x"))
        """
        theta = np.pi / 2
        Rx = np.array(
            [
                [self.zoom, 0, 0, 0],
                [0, self.zoom * np.cos(theta), self.zoom * np.sin(theta), 0],
                [0, -self.zoom * np.sin(theta), self.zoom * np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [self.zoom * np.cos(theta), 0, -self.zoom * np.sin(theta), 0],
                [0, self.zoom, 0, 0],
                [self.zoom * np.sin(theta), 0, self.zoom * np.cos(theta), 0],
                [0, 0, 0, 1],
            ]
        )
        Rz = np.array(
            [
                [self.zoom * np.cos(theta), -self.zoom * np.sin(theta), 0, 0],
                [self.zoom * np.sin(theta), self.zoom * np.cos(theta), 0, 0],
                [0, 0, self.zoom, 0],
                [0, 0, 0, 1],
            ]
        )
        if direction == "x":
            self.view._set_camera_orientation([x for x in Rx.flatten()])
        if direction == "y":
            self.view._set_camera_orientation([x for x in Ry.flatten()])
        if direction == "z":
            self.view._set_camera_orientation([x for x in Rz.flatten()])

    def replace_trajectory(self, *args, traj, representation="ball+stick"):

        self.traj = traj
        comp_ids = []
        orientation_ = self.view._camera_orientation
        orientation = [x for x in orientation_]
        # Camera view is empty before viewing
        if orientation:
            # Keep last number to 1
            orientation.pop()
            orientation.append(1)
        self.removeArrows()
        # Remove all components except the newly added one
        for comp_id in self.view._ngl_component_ids:
            comp_ids.append(comp_id)

        self.view.add_trajectory(self.traj)

        for comp_id in comp_ids:
            self.view.remove_component(comp_id)

        self.view._set_camera_orientation(orientation)
        self.modifiy_representation()
        self.addArrows()
        # self.view.control.zoom(0.75)
    def modifiy_representation(self,*args):
        self.view.clear_representations()
        if self.representation=="spacefill":
            self.view.add_representation(
                self.representation, selection="all", radius=self.slider_atom_radius.value
            )
        elif self.representation=="ball+stick":
            self.view.add_representation(
                self.representation, selection="all", radius=self.slider_atom_radius.value, aspectRatio=self.slider_aspect_ratio.value
            )
    def removeArrows(self):
        self.view._execute_js_code(
            """
        this.stage.removeComponent(this.stage.getComponentsByName("my_shape").first)
        """
        )

    def addArrows(self, *args):
        self.removeArrows()

        positions = list(self.traj[0].get_positions().flatten())
        delta = (self.traj[1].get_positions() - self.traj[0].get_positions()).T
        delta[(delta < 1e-4) & (delta > -1e-4)] = 0
        directions = delta / np.linalg.norm(delta, axis=0)
        directions = np.nan_to_num(
            directions
        )  # If vector does not move, replace nan with 0
        n_atoms = int(len(positions) / 3)
        color = n_atoms * [0, 1, 0]
        radius = n_atoms * [self.slider_arrow_radius.value]
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

        freq = (self.view.max_frame + 1) / 4
        sinus = np.sin(np.arange(0, (self.view.max_frame + 1)) / freq * np.pi / 2)
        # Remove observe callable to avoid visual glitch
        if self.handler:
            self.view.unobserve(self.handler.pop(), names=["frame"])

        def on_frame_change(change):
            frame = change["new"]

            positions = self.traj[frame].get_positions().flatten()
            n_atoms = int(len(positions) / 3)
            step = np.array(
                [
                    x * sinus[frame] * self.slider_amp_arrow.value
                    for j in range(n_atoms)
                    for x in directions[:, j]
                ]
            )
            if self.tick_box.value:
                positions2 = list(positions + step)
                positions = list(positions)
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
                positions = list(positions)
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

    def set_view_dimensions(self, width=640, height=480):
        """
        Set view width and height in px
        """
        self.view._remote_call(
            "setSize", target="Widget", args=[str(width) + "px", str(height) + "px"]
        )

    def set_player_parameters(self, **kargs):
        """
        Available parameters :
            step : Initial value = 1
                Available values [-100, 100]
            delay : Initial value = 100
                Available values : [10, 1000]
        """
        self.view.player.parameters = kargs

    def set_view_parameters(self, **kwargs):
        """
        Available parameters:
            panSpeed : Initial value = 1
                Available values : [0, 10]
            rotateSpeed : Initial value = 2
                Available values : [0, 10]
            zoomSpeed : Initial value = 1
                Available values : [0, 10]
            clipDist : Initial value = 10
                Available values : [0, 200]
            cameraFov : Initial value = 40
                Available values : [15, 120]
            clipFar : Initial value = 100
                Available values : [0, 100]
            clipNear : Initial value = 0
                Available values : [0, 100]
            fogFar : Initial value = 100
                Available values : [Available values]
            fogNear : Initial value = 50
                Available values : [0, 100]
            lightIntensity : Initial value = 1
                Available values : [0, 10]
            quality : Initial value = 'medium'
                Available values : 'low', 'medium', 'high'
            backgroundColor : Initial value = 'white'
                Available values : color name or HEX
        
        """
        self.view.parameters = kwargs

    def change_resolution(self, *args):
        # We use half value, because it doubles when we download
        if self.dropdown_resolution.value == "480p":
            self.set_view_dimensions(320, 240)
        elif self.dropdown_resolution.value == "720p":
            self.set_view_dimensions(640, 360)
        elif self.dropdown_resolution.value == "1080p":
            self.set_view_dimensions(960, 540)
        elif self.dropdown_resolution.value == "1440p":
            self.set_view_dimensions(1280, 720)
        elif self.dropdown_resolution.value == "2K":
            self.set_view_dimensions(1024, 540)
        elif self.dropdown_resolution.value == "500x500p":
            self.set_view_dimensions(250, 250)
        elif self.dropdown_resolution.value == "1000x1000p":
            self.set_view_dimensions(500, 500)

    def make_movie(self, *args):
        # Remove gif preview
        self.output_gif.outputs=()
        self.button_download.layout.visibility = "hidden"
        # Stop animation
        self.view._iplayer.children[0]._playing = False
        # Set resolution
        self.change_resolution()
        thread = threading.Thread(target=self.process_images,)
        thread.daemon = True
        thread.start()
        # thread.join()
        # if self.flag_movie:
        #     self.show_gif_preview()
    def process_images(self, *args):
        # Disable widgets to avoid glitch in video
        for widget in self.widgetList:
            widget.disabled = True
        n_frames = self.view.max_frame + 1
        self.output_movie.outputs=()
        self.output_movie.append_stdout("Generating GIF, please wait...")

        tmp_dir_frames = tempfile.TemporaryDirectory(prefix="frames_", dir=self.tmp_dir.name)
        try:
            for frame in range(n_frames):
                counter = 0
                im = self.view.render_image(frame=frame, factor=2)

                while not im.value:
                    time.sleep(0.1)
                    counter += 1
                    if counter > 50:
                        self.output_movie.outputs=() 
                        self.output_movie.append_stdout("Could not generate pictures")
                        raise Exception("Could not generate pictures")
                path = os.path.join(tmp_dir_frames.name, f"frame{frame}.png")
                with open(path, "wb") as f:
                    f.write(im.value)

            # Resets view dimensions to original
            self.set_view_dimensions()

            self.compile_movie(directory=tmp_dir_frames.name, n_frames=n_frames)
            self.show_gif_preview()
            # self.button_download.layout.visibility = "visible"
            
        except Exception as error:
            self.output_movie.clear_output()
            self.output_movie.append_stdout(error)
        finally:
            for widget in self.widgetList:
                widget.disabled = False
            # Resets view dimension to original
            self.set_view_dimensions()
            
        tmp_dir_frames.cleanup()
        
        # self.output_movie.clear_output() # NOT WORKING
        self.output_movie.outputs=() 
        # self.output_movie.append_stdout(50*" "+"\r")
        self.output_movie.append_stdout('Right click on GIF to download it')

    def compile_movie(self, *args, directory, n_frames):

        imagefiles = [
            os.path.join(directory, f"frame{i}.png") for i in range(0, n_frames)
        ]
        frame_per_second = round(1000 / (25 / self.slider_speed.value))
        im = mpy.ImageSequenceClip(imagefiles, fps=frame_per_second)
        with tempfile.NamedTemporaryFile(
            dir='.', prefix="movie_", suffix=".gif", delete=False
        ) as tmpFile:
            tmpFileName_movie = os.path.basename(tmpFile.name)
            self.tmpFileName_movie=os.path.join(self.tmp_dir.name,tmpFileName_movie)
            im.write_gif(self.tmpFileName_movie, fps=frame_per_second, verbose=False, logger=None)
        # shutil.move(tmpFileName_movie,self.tmpFileName_movie)
        # self.change_movie()
        # self.view._iplayer.children[0]._playing = True

    def change_movie(self, *args):

        with open( self.tmpFileName_movie, "rb") as f:
            res = f.read()
            filename = "movie.gif"
            b64 = base64.b64encode(res)
            payload = b64.decode()

        html_code = """<html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
        <a download="{filename}" href="data:image;base64,{payload}" download>
        <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download File</button>
        </a>
        </body>
        </html>
        """
        html_code = html_code.format(payload=payload, filename=filename)
        self.button_download.value = html_code

        os.remove(self.tmpFileName_movie)

    def hide_button(self, *args):
        self.button_download.layout.visibility = "hidden"

    def set_speed(self, *args):
        self.view.player.update_parameters(
            change={"new": {"delay": self.init_delay / self.slider_speed.value}}
        )

    def show_gif_preview(self,*args):
        val= self.dropdown_resolution.value
        if val == "480p":
            width, height = 400,300
        elif val == "720p" or val=='1080p' or val=='1440p' or val=='2K':
            width, height = 400, 225
        elif val =='500x500p' or val=='1000x1000p':
            width, height = 400, 400
        with open(self.tmpFileName_movie ,'rb') as f:
            gif_bytes=f.read()
            gif=Image(data=gif_bytes, format='gif',width=width,height=height)
        self.output_gif.append_display_data(gif)

        self.show_download_button(gif_bytes)
        
    
    def show_download_button(self,gif_bytes):
        b64 = base64.b64encode(gif_bytes)
        payload = b64.decode()
        filename="movie.gif"
        html_code=f'<a href="data:image/png;base64,{payload}" > Download GIF</a>' # width="{width}" height="{height}"
        self.gif_link.value=html_code
        html_code = f"""<html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        </head>
        <body>
        <a download="{filename}" href="data:image;base64,{payload}" download>
        <button class="p-Widget jupyter-widgets jupyter-button widget-button mod-warning">Download File</button>
        </a>
        </body>
        </html>
        """
        
        # html_code = html_code.format(payload=payload, filename=filename)
        # self.button_download.value = html_code
        # self.button_download.layout.visibility = "visible"
        # self.remove_movie() # Does not work

    def remove_movie(self,*args):
        if self.tmpFileName_movie:
            os.remove(self.tmpFileName_movie)
            self.tmpFileName_movie=None