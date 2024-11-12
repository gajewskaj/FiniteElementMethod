import os
import gmshparser

mesh = gmshparser.parse(os.path.join("input", "quadrilateral_mesh.msh"))
print(mesh)

# for entity in mesh.get_node_entities():
#     for node in entity.get_nodes():
#         nid = node.get_tag()
#         ncoords = node.get_coordinates()
#         print("Node id = %s, node coordinates = %s" % (nid, ncoords))

for entity in mesh.get_element_entities():
    eltype = entity.get_element_type()
    print("Element type: %s" % eltype)
    for element in entity.get_elements():
        elid = element.get_tag()
        elcon = element.get_connectivity()
        print("Element id = %s, connectivity = %s" % (elid, elcon))