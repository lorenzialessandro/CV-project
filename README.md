# CV-project

## Introduction
The following folder contains the code of the computer vision project that focuses on several critical tasks related to **motion capture**. First, we examine standard output files from a motion capture system, visualizing human skeletons and rigid bodies in 3D using Python. We then tackle the problem of flickering in rigid body motion caused by marker occlusions, using techniques such as the Kalman filter and the Particle filters. Finally, we integrate motion capture data into Unreal Engine to animate a character and project the skeleton data onto a 2D image plane, applying the necessary geometry.

### Usage

```
git clone https://github.com/lorenzialessandro/CV-project.git
cd CV-project
```

## Task 1
We have employed a structured approach to read, store and visualize data from CSV, C3D, and BVH files, each of which containing different motion capture data elements.

**CSV files**: we extracted the x, y, and z coordinates for each joint of the skeleton, along with their connections and we plot the data in 3D, showing the relative positions of various bones and joints (such as the head, neck, shoulders, …) and also both the connections along the markers and the trajectories of the skeleton. Similar to the skeleton, we read the rigid body data using a function to extract the marker positions, visualizing in 3D the markers and their connections.

|Skeleton from CSV file | Rigid Body from CSV file|
|-|-|
|![](./media/skeleton.gif)|![](./media/ragnetto-basic.gif)|



In the **BVH file** we parsed the file to retrieve the relative rotations, global positions, edges (bone connections), offsets (bone lengths), and joint names. For the **C3D file** we read the file and extracted a tuple containing a list of dictionaries with the marker information for each frame and the labels: a list of marker labels.

### Usage

```
python3 main1.py <filter_type>
```
\<filter_type\> : mandatory parameter, chose among : { CSV_SKELETON | CSV_RIGID | BVH | C3D}
- CSV_SKELETON : read the [skeleton.csv](./resources/360fps/skeleton.csv) and plot the corresponding skeleton animation. It's also possible to create the `.gif` file using the `create_animation` function. 
- CSV_RIGID : read the [rigidbody.csv](./resources/360fps/rigidbody.csv) and plot the corresponding rigid body animation. It's also possible to create the `.gif` file using the `create_animation` function.
- BVH : read the [animation.bvh](./resources/360fps/animation.bvh), store and print the corresponding information in variables. It's also possible to store them in a `.txt` file using the `write_bvh_info_to_txt` function. 
- C3D : read the [marker.c3d](./resources/360fps/marker.c3d), store and print the corresponding information in variables. It's also possible to store them in a `.txt` file using the `write_c3d_info_to_txt` function. 



## Task 2

To address the issue of flickering in the motion of the rigid body due to occlusions, two different filtering methods were implemented: **Kalman Filter** (KF) and **Particle Filter** (PF).

|KF on Rigid Body|PF on Rigid Body|
|-|-|
|![](./media/ragnetto-KF.gif)|![](./media/ragnetto-PF.gif)|

### Usage
```
python3 main2.py <filter_type> <filter_type>
```
\<filter_type\> : mandatory parameter, chose among : { CSV_SKELETON | CSV_RIGID}
\<filter_type> : mandatory parameter, chose among : { KF | PF}

- KF : apply the **Kalman Filter** to the csv input file and create the corresponding corrected animation. It's also possible to create the `.gif` file using the `create_animation` function.
- PF : apply the **Particle Filter** to the csv input file and create the corresponding corrected animation. It's also possible to create the `.gif` file using the `create_animation` function.



## Task 3

Inside an <b>Unreal Engine 5 (UE5)</b> virtual environment we want to achieve 3D to 2D projection of joint positions onto the camera plane.
We properly modelled the scene inserting 2 core blueprints, one containing our main actor and the other containing the camera
<br>
<p align="center">
  <img src="media/scene.png" width="75%">
</p>

Using the blueprint engine together with [json Blueprint Utilities](https://www.unrealdirective.com/tips/json-blueprint-utilities-plugin) plugin we implemented a script to extract data in <b>Json</b> format (available for visualization [**Here**](https://blueprintue.com/blueprint/_qn_vgvc/)). Using openCV and the extracted data we're able to project skeleton joints on the camera frame:
<br>
### Usage
```
python3 main_3.py
```
<br>
<p align="center">
  <img src="media/skeletonProjection.png" width="75%">
</p>
<br>

## Extra Task
Instead of evaluating results on Matplotlib or open3d it would be much better to forward data to a more suitable environment such as Blender.
We adapted [**deep-motion-editing**](https://github.com/DeepMotionEditing/deep-motion-editing) framework to our data.
Please use as reference requirements specified in the deep-motion-editing repository.
<br><br>

before testing, set an alias to the blender executable as :
(For Linux users)
```
gedit ~/.bashrc
```
and write as last line of file
```
export PATH=/path/to/blender/folder/:$PATH
```
Then save and close. From now on running "blender" on the terminal directly launch the blender environment



### Usage

```
python3 main_extra.py <OPT>
```
\<OPT\> : mandatory parameter, chose among : { RENDER | SKINNING }
* RENDER : imports the [animation_small.bhv](resources/360fps/animation_small.bvh) into a Blender scene, adding : checkerboard floor, sun, meshes linked to bones and assigned to a default materials + a camera ready to render the scene
* SKINNING : imports the [animation.bhv](resources/360fps/animation.bvh) as animation and retargets it to a [mesh](resources/ue5/TheBoss.fbx)
<br>
<p align="center">
  <img src="media/bvhFrame.png" width="75%">
</p>

---

## Conclusions
We have successfully solved all the presented tasks related to the field of motion capture. In particular, we first analyzed and plotted human skeletons and rigid bodies in 3D, reading data from different types of input files. Then, by implementing both the Kalman filter and the Particle filter, we solved the problem of flickering due to occlusion. Finally, we interacted with Unreal Engine 5 to get the joints projected from 3D to 2D. We also connected our data to the Blender environment. 

