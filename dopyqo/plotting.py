import numpy as np
import pyvista as pv
from dopyqo.colors import *
from dopyqo.xsf import load_xsf


def plot_3d_data(
    coordinates: np.ndarray,
    data: np.ndarray,
    isosurfaces: int | list[float],
    isovals_as_percent: bool = False,
    atom_positions: np.ndarray | None = None,
    atom_labels: list[str] | None = None,
    extend_data: bool = False,
    grid_vectors: np.ndarray | None = None,
    cmap: str = "coolwarm",
    atom_color: str = "#0dff00",
    atom_radius: float = 0.3,
    opacity: float = 1.0,
    filtered_regions: int = 0,
    point_of_interest: np.ndarray = np.array([0.0, 0.0, 0.0]),
    html_filename: str = "3d_pyvista.html",
    plotter: pv.Plotter | None = None,
    meshes: list[pv.PolyData | pv.DataSet | pv.MultiBlock | np.ndarray] = [],
):
    """Plot 3D data given by coordinates and scalar values for each coordinate (data).
    The coordinates need to be structured in a way that we can create a PyVista/VTK StructuredGrid from them.
    The data does not need to be defined on a orthogonal grid. See the definition/usage of PyVista/VTK StructuredGrid for more information.
    At the end the plot will be saved to a html-file.

    Args:
        coordinates (np.ndarray): Numpy array of coordinates of shape (nx,ny,nz,3) or of shape (nx*ny*nz,3)
        data (np.ndarray): Numpy array of scalar field data of shape (nx,ny,nz) or of shape (nx*ny*nz)
        isosurfaces (int | list[float]): isosurfaces argument passed to PyVista mesh.contour function
        isovals_as_percent (bool): Should the isosurface values (isosurfaces argument) be interpreted as percentage value relative to
                                   the minimum and maximum of the data values. Can only be set to true if isosurfaces argument is a list.
        atom_positions (np.ndarray | None, optional): Numpy array of atom positions of shape (N,3). If None no atoms will be plotted.
                                                      Defaults to None.
        extend_data (bool): Whether to extend the data along each grid vector once. If True grid_vectors has be given as numpy array. Defaults to False.
        grid_vectors (np.ndarray | None): Numpy array of vectors defining the grid, e.g. lattice vectors, of shape (3,3), where each row is one vector.
                                          Has not to be numpy array if extend_data is True. Defaults to None.
        cmap (str, optional): cmap argument passed to PyVista Plotter.add_mesh function for plotting the isosurfaces. Defaults to "coolwarm".
        atom_color (str, optional): color argument passed to PyVista Plotter.add_mesh for plotting the atom spheres. Defaults to "#0dff00".
        opacity (float, optional): Opacity value of the shown isosurfaces. Defaults to 1.0.
        filtered_regions (int, optional): Number of isosurfaces around point_of_interest that are highlighted while all others are plotted with opacity*0.2.
                                          Can be slow. If set to 0 all isosurfaces are plotted. Defaults to 0.
        point_of_interest (np.ndarray, optional): point_of_interest used for filtering isosurfaces. See filtered_regions argument. Defaults to np.array([0.0, 0.0, 0.0]).
        html_filename (str, optional): Filename of the html-file the plot will be saved to. Postfix ".html" is appended if not already present. Defaults to "3d_pyvista.html".
        plotter (pv.Plotter | None, optional): PyVista Plotter object used for plotting. If None a new object will be created. Defaults to None.
        meshes (list[pv.PolyData | pv.DataSet | pv.MultiBlock | np.ndarray]): List of additional meshes that are plotted. Are each passed to plotter.add_mesh function.
                                                                              Defaults to [].
    """
    if not html_filename.endswith(".html"):
        html_filename += ".html"
    if atom_labels is None and atom_positions is not None:
        atom_labels = ["" for _ in atom_positions]
    if isovals_as_percent and not isinstance(isosurfaces, list):
        print(
            f"{ORANGE}Plotting warning: Argument isovals_as_percent is set to true but argument isosurfaces is not a list. Setting isovals_as_percent to False!{RESET_COLOR}"
        )
        isovals_as_percent = False

    if isovals_as_percent:
        isosurfaces = [data.min() + x * data.ptp() for x in isosurfaces]

    nx, ny, nz = data.shape

    mesh = pv.StructuredGrid()
    mesh.points = coordinates.reshape((-1, 3))
    # If dimensions are not specified the mesh remains "empty" and cannot be plotted
    mesh.dimensions = [nx, ny, nz]
    mesh.point_data["values"] = data.ravel()
    contours = mesh.contour(isosurfaces=isosurfaces)

    extended_contours = []
    if extend_data:
        if grid_vectors is None:
            print(
                f"{RED}Plot warning: extend_data is set to True but grid_vectors are not given. Cannot extend data without grid vectors.{RESET_COLOR}"
            )
        else:
            coordinates_list_tmp = []
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 0]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in first grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 1]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in second grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 2]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in third grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 0]
            coordinates_tmp += grid_vectors[None, 1]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in first + second grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 0]
            coordinates_tmp += grid_vectors[None, 2]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in first + third grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 1]
            coordinates_tmp += grid_vectors[None, 2]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in second + third grid direction
            coordinates_tmp = coordinates.copy()
            coordinates_tmp += grid_vectors[None, 0]
            coordinates_tmp += grid_vectors[None, 1]
            coordinates_tmp += grid_vectors[None, 2]
            coordinates_list_tmp.append(coordinates_tmp)  # shifted in first + second + third grid direction

            for coordinates_tmp in coordinates_list_tmp:
                mesh_tmp = pv.StructuredGrid()
                mesh_tmp.points = coordinates_tmp.reshape((-1, 3))
                mesh_tmp.dimensions = [nx, ny, nz]
                mesh_tmp.point_data["values"] = data.ravel()
                contours_tmp = mesh_tmp.contour(isosurfaces=isosurfaces)
                extended_contours.append(contours_tmp)

    ##############################################################
    #          Define PyVista spheres for atom positions         #
    ##############################################################
    # NOTE: Point cloud would also be possible but did not render as spheres on our machine
    spheres = []
    if atom_positions is not None:
        spheres = [pv.Sphere(radius=atom_radius, center=pos) for pos in atom_positions]

    ##############################################################
    #                     Filter isosurfaces                     #
    ##############################################################
    if filtered_regions != 0:
        if extend_data:
            print(
                f"{ORANGE}Plot warning: You specified to filter regions ({filtered_regions=}) and to extend the data ({extend_data=}). "
                + f"Beware that currently isosurfaces in the extend data are not filtered.{RESET_COLOR}"
            )
        all_regions = contours.connectivity("all")
        region_ids = np.unique(all_regions["RegionId"])
        my_region_ids = region_ids[:filtered_regions]
        my_regions = contours.connectivity("specified", my_region_ids)
        # NOTE: Calculating distances like in the following takes quite long. How to speed up?
        distances = [
            np.min(np.linalg.norm(contours.connectivity("specified", region_id).points - point_of_interest, axis=1)) for region_id in region_ids
        ]
        my_region_ids = np.argsort(distances)[:2]
        my_regions = contours.connectivity("specified", my_region_ids)

        largest_region = contours.connectivity("largest")

    ##############################################################
    #                          Plotting                          #
    ##############################################################
    # contours = mesh.contour(isosurfaces=7)
    if plotter is None:
        plotter = pv.Plotter(off_screen=True)
    plotter.add_mesh(mesh.outline())
    if filtered_regions != 0:
        plotter.add_mesh(contours, show_edges=False, cmap=cmap, opacity=opacity * 0.2)
        plotter.add_mesh(my_regions, show_edges=False, cmap=cmap, opacity=opacity)
    else:
        plotter.add_mesh(contours, show_edges=False, cmap=cmap, opacity=opacity, label="data")
    for contour_tmp in extended_contours:
        plotter.add_mesh(contour_tmp, show_edges=False, cmap=cmap, opacity=opacity)
    for sphere_tmp, label_tmp in zip(spheres, atom_labels):
        label_tmp = label_tmp.strip()
        plotter.add_mesh(sphere_tmp, color=atom_color)
        text_height = 0.2 * len(label_tmp.splitlines())
        text_depth = text_height / 5.0
        text_tmp = pv.Text3D(label_tmp, center=[0, 0, 0], height=text_height, depth=text_depth)
        width_tmp = text_tmp.bounds.x_max - text_tmp.bounds.x_min
        height_tmp = text_tmp.bounds.y_max - text_tmp.bounds.y_min
        label_pos = np.array(sphere_tmp.center) + np.array([atom_radius + width_tmp / 2, atom_radius + height_tmp / 2, 0.0])
        text_tmp = pv.Text3D(label_tmp, center=label_pos, height=text_height, depth=text_depth, normal=[0, 0, 1])
        plotter.add_mesh(text_tmp, color="black")
    for mesh_tmp in meshes:
        plotter.add_mesh(mesh_tmp)
    # text_tmp = pv.Text3D("Test text", center=(-10.0, 0.0, 10.0))
    # plotter.add_mesh(text_tmp)
    # # Label not showing in interactive/html view, see:
    # # https://github.com/pyvista/pyvista/issues/6202
    # label = pv.Label(text="Test text", position=(-10.0, 0.0, 10.0), size=50)
    # plotter.add_actor(label)
    plotter.show_grid()
    plotter.show_axes()
    plotter.export_html(html_filename)
    print(f"{GREEN}Plot sucessfully saved to {html_filename}.{RESET_COLOR}")

    # NOTE: Volume plot only works for numpy array, ImageData, RectilinearGrid or UnstructuredGrid
    # plotter.add_volume(xsf_data, scalars="values", cmap="magma", opacity=[1,0,1])
    # plotter.add_volume(mesh, scalars="orbital", cmap="magma", opacity=[1,0,1])

    # # Plot mesh itself without the scalar data associated to each grid point
    # # This shows the shape of the unit cell and each voxel in the dataset
    # plotter = pv.Plotter(off_screen=True)
    # plotter.add_mesh(mesh)
    # plotter.add_points(mesh.points, color="red", point_size=2)
    # plotter.add_mesh(
    #     mesh.get_cell(mesh.n_cells - 1).cast_to_unstructured_grid(),
    #     color="pink",
    #     edge_color="blue",
    #     line_width=5,
    #     show_edges=True,
    # )
    # plotter.show_grid()
    # plotter.show_axes()
    # plotter.export_html("contour_plot.html")

    return plotter


def plot_xsf(
    filename: str,
    isosurfaces: int | list[float],
    isovals_as_percent: bool = False,
    cmap: str = "coolwarm",
    color_atoms: str = "#0dff00",
    opacity: float = 1.0,
    filtered_regions: int = 0,
    point_of_interest: np.ndarray = np.array([0.0, 0.0, 0.0]),
    plot_lattice_vectors: bool = True,
    html_filename: str = "3d_pyvista.html",
    plotter: pv.Plotter | None = None,
    plot_abs: bool = False,
):
    xsf_data, xsf_grid_vecs, xsf_origin, xsf_atom_pos, xsf_primvecs, xsf_convvecs = load_xsf(filename=filename)
    if plot_abs:
        xsf_data = np.abs(xsf_data)
    # Each row of xsf_grid_vecs is one grid_vector

    ##############################################################
    #              Calculate data point coordinates              #
    ##############################################################
    nx, ny, nz = xsf_data.shape
    coordinates = (
        np.arange(nx)[:, None] * xsf_grid_vecs[0] / nx
        + np.arange(ny)[:, None, None] * xsf_grid_vecs[1] / ny
        + np.arange(nz)[:, None, None, None] * xsf_grid_vecs[2] / nz
        + xsf_origin
    )

    lattice_vectors_meshes = []
    if plot_lattice_vectors:
        lattice_vectors_meshes = [pv.Arrow(start=xsf_origin, direction=v, shaft_radius=2e-2, tip_radius=5e-2, scale="auto") for v in xsf_grid_vecs]

    from dopyqo.plotting import plot_3d_data

    return plot_3d_data(
        coordinates=coordinates,
        data=xsf_data,
        isosurfaces=isosurfaces,
        isovals_as_percent=isovals_as_percent,
        atom_positions=xsf_atom_pos,
        meshes=lattice_vectors_meshes,
        cmap=cmap,
        atom_color=color_atoms,
        opacity=opacity,
        filtered_regions=filtered_regions,
        point_of_interest=point_of_interest,
        html_filename=html_filename,
        plotter=plotter,
    )
