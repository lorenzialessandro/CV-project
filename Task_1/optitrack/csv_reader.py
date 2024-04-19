"""\
optitrack.csv_reader : a plain-Python parser for reading Optitrack CSV files in version 1.2 or 1.21 format.

This uses only Python modules common between CPython, IronPython, and
RhinoPython for compatibility with both Rhino and offline testing.

Reference for the format: http://wiki.optitrack.com/index.php?title=Data_Export:_CSV

Note that the file format has changed significantly since version 1.1.

Copyright (c) 2016, Garth Zeglin. All rights reserved. Licensed under the
terms of the BSD 3-clause license as included in LICENSE.

"""
################################################################
# Rhino IronPython does not include the csv module, so a very simple
# implementation of a csv file reader is included.  This only supports the
# limited use of csv as generated by the Optitrack Motive software.

import csv
class CSVReader(object):
    
    def __init__(self, stream):
        self._stream = stream
        return

    def __iter__(self):
        return self
    
    def next(self):

        # Read the next raw line from the input.
        line = self._stream.next().rstrip()

        # Make sure than empty lines are returned as empty lists.
        if line == '':
            return list()
        
        # Note: following is a very format-specific hack.  Currently, the only
        # quoted fields are the ID fields, which don't really need them.  This
        # quick trick means that commas are not allowed in any body names.

        # Remove all quoting marks:
        unquoted = line.replace('"','')

        # And then just use split to separate fields based on commas.
        return unquoted.split(',')
        
################################################################
# define a utility object for describing the mapping from CSV columns to data objects
import collections
ColumnMapping = collections.namedtuple('ColumnMapping', ['setter', 'axis', 'column'])

################################################################
class RigidBody(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.rotations = list()  # list with one element per frame, either None or [x,y,z,w] float lists
        self.times     = list()  # list with one element per frame with the capture time   
        
        self.rigid_body_markers = dict()
        self.markers = dict()
                 
        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        self.rotations.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:  
                self.positions[frame] = [0.0,0.0,0.0]                
            self.positions[frame][axis] = float(value)  


    def _set_rotation( self, frame, axis, value ):
        if value != '':
            if self.rotations[frame] is None:
                self.rotations[frame] = [0.0,0.0,0.0,0.0]
            self.rotations[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count
    
class Skeleton(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.rotations = list()  # list with one element per frame, either None or [x,y,z,w] float lists
        self.times     = list()  # list with one element per frame with the capture time   
        
        self.bones = dict()
        self.bone_markers = dict()
                 
        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        self.rotations.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:  
                self.positions[frame] = [0.0,0.0,0.0]                
            self.positions[frame][axis] = float(value)  


    def _set_rotation( self, frame, axis, value ):
        if value != '':
            if self.rotations[frame] is None:
                self.rotations[frame] = [0.0,0.0,0.0,0.0]
            self.rotations[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count

    
################################################################

class RigidBodyMarker(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.times     = list()  # list with one element per frame with the capture time   
        self.quality = 0.0

        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:
                self.positions[frame] = [0.0,0.0,0.0] 
            self.positions[frame][axis] = float(value)

    def _set_quality( self, frame, value):
        if value != '':
            if self.quality[frame] is None:
                self.quality[frame] = 0.0
            self.quality[frame] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count

################################################################

class Marker(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.times     = list()  # list with one element per frame with the capture time   

        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:
                self.positions[frame] = [0.0,0.0,0.0]           
            self.positions[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count

###################################################################

class Bone(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.rotations = list()  # list with one element per frame, either None or [x,y,z,w] float lists
        self.times     = list()  # list with one element per frame with the capture time
                
        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        self.rotations.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:  
                self.positions[frame] = [0.0,0.0,0.0]                
            self.positions[frame][axis] = float(value)  


    def _set_rotation( self, frame, axis, value ):
        if value != '':
            if self.rotations[frame] is None:
                self.rotations[frame] = [0.0,0.0,0.0,0.0]
            self.rotations[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count

################################################################

class BoneMarker(object):
    """Representation of a single rigid body."""

    def __init__(self, label, ID):
        self.label     = label
        self.ID        = ID
        self.positions = list()  # list with one element per frame, either None or [x,y,z] float lists
        self.times     = list()  # list with one element per frame with the capture time   

        return

    def _add_frame(self, t):
        self.times.append(t)
        self.positions.append(None)
        
    def _set_position( self, frame, axis, value ):
        if value != '':
            if self.positions[frame] is None:
                self.positions[frame] = [0.0,0.0,0.0]           
            self.positions[frame][axis] = float(value)

    def num_total_frames(self):
        return len(self.times)

    def num_valid_frames(self):
        count = 0
        for pt in self.positions:
            if pt is not None:
                count = count + 1
        return count


################################################################


class Take(object):
    """Representation of a motion capture Take.  Each CSV file represents one Take.
    """

    def __init__(self):

        # user-accessible properties
        self.frame_rate    = 120.0
        self.rotation_type = 'Quaternion'
        self.units         = 'Meters'

        # user-accessible data
        self.rigid_bodies = dict()      # dict of RigidBody objects, indexed by asset name string
        self.skeletons = dict()
        
        # raw header information is saved as follows:
        self._raw_info    = dict()      # line 1: raw header fields, with values as unparsed strings
        self._raw_types   = list()      # line 3: raw column types for all data columns (not including frame and time column)
        self._raw_labels  = list()      # line 4: raw asset names for all data columns (not including frame and time column)
        self._raw_fields  = list()      # line 6: raw field types for all data columns (not including frame and time column)
        self._raw_axes    = list()      # line 7: raw axis designators for all data columns (not including frame and time column)
        self._ignored_labels  = set()   # names of all ignored objects
        self._column_map = list()       # list of ColumnMap tuples defining where to store data column elements
        
        return

    def readCSV(self, path, verbose=False):
        """Load a CSV motion capture data file."""

        self.rigid_bodies = dict()
        self.skeletons = dict()
        
        self._raw_info = dict()
        self._ignored_labels  = set()
        self._column_map = list()

        csv_stream = csv.reader(open(path, "r"))
        self._read_header(csv_stream, verbose)
        self._read_data(csv_stream, verbose)        
        
        return self

    # ================================================================
    def _read_header(self, stream, verbose = False):

        # Line 1 consists of a series of token, value pairs.
        line1 = next(stream)
        assert line1[0] == 'Format Version', "Unrecognized header cell: %s" % line1[0]
        format = line1[1]
        
        assert format == '1.23' or format == '1.2', "Unsupported format version: %s" % line1[1]

        for columnidx in range(int(len(line1)/2)):
            self._raw_info[ line1[2*columnidx]] = line1[2*columnidx+1]

        # make a few assumptions about data type
        self.rotation_type = self._raw_info.get('Rotation Type')
        assert self.rotation_type == 'Quaternion', 'Only the Quaternion rotation type is supported, found: %s.' % self.rotation_type
        
        # pull a few values out, supplying reasonable defaults if they are missing
        self.frame_rate = float(self._raw_info.get('Export Frame Rate', 120))
        self.units = self._raw_info.get('Length Units', 'Meters')

        # Line 2 is blank.
        line2 = next(stream)
        assert len(line2) == 0, 'Expected blank second header line, found %s.' % line2

        # Line 3 designates the data type for each succeeding column.
        line3 = next(stream)
        self._raw_types = line3[2:]

        # check for any unexpected types on line 3
        all_types = set( self._raw_types )
        supported_types = set(['Rigid Body', 'Rigid Body Marker', 'Marker','Bone', 'Bone Marker'])
        assert all_types.issubset(supported_types), 'Unsupported object type found in header line 3: %s' % all_types

        # Line 4 designates the asset labels for each column (e.g. 'Rigid Body 1', or whatever name was assigned)
        line4 = next(stream)
        self._raw_labels = line4[2:]

        # Line 5 designates the marker ID for each column
        line5 = next(stream)
        
        # Line 6 designates the data type for each column: Rotation, Position, Error Per Marker, Marker Quality
        line6 = next(stream)
        self._raw_fields = line6[2:]
        
        # Line 7 designates the specific axis: Frame, Time, X, Y, Z, W, or blank
        line7 = next(stream)
        self._raw_axes = line7[2:]
        
        # Process lines 3-7 at once, creating named objects to receive each frame of data for supported asset types.
        for col,asset_type,label,ID,field,axis in zip( range(len(self._raw_types)), self._raw_types, self._raw_labels, \
                                                             line5[2:], self._raw_fields, self._raw_axes ):
            if asset_type == 'Rigid Body' :
                if label in self.rigid_bodies:
                    body = self.rigid_bodies[label]
                else:
                    body = RigidBody(label,ID)
                    self.rigid_bodies[label] = body

                # create a column map entry for each rigid body axis
                if field == 'Rotation':
                    axis_index = {'X':0, 'Y':1, 'Z':2, 'W': 3}[axis]
                    setter = body._set_rotation
                    self._column_map.append(ColumnMapping(setter, axis_index, col))

                elif field == 'Position':
                    axis_index = {'X':0, 'Y':1, 'Z':2}[axis]
                    setter = body._set_position
                    self._column_map.append(ColumnMapping(setter, axis_index, col))
            
            # Extract bone informations
            elif asset_type == 'Bone':
                root = label.split(":")[0]
                if root not in self.skeletons:
                    body = Skeleton(root,ID)
                    self.skeletons[root] = body
                    
                if label in self.skeletons:
                    body = self.skeletons[label]
                else:
                    body = Bone(label,ID)
                    self.skeletons[label] = body

                # create a column map entry for each rigid body axis
                if field == 'Rotation':
                    axis_index = {'X':0, 'Y':1, 'Z':2, 'W': 3}[axis]
                    setter = body._set_rotation
                    self._column_map.append(ColumnMapping(setter, axis_index, col))

                if field == 'Position':
                    axis_index = {'X':0, 'Y':1, 'Z':2}[axis]
                    setter = body._set_position
                    self._column_map.append(ColumnMapping(setter, axis_index, col))
                        # Extract bone informations
                        
            elif asset_type == 'BoneMarker':
                root = label.split(":")[0]
                if root not in self.skeletons:
                    body = Skeleton(root,ID)
                    self.skeletons[root] = body
                    
                if label in self.skeletons:
                    body = self.skeletons[label]
                else:
                    body = BoneMarker(label,ID)
                    self.skeletons[label] = body

                if field == 'Position':
                    axis_index = {'X':0, 'Y':1, 'Z':2}[axis]
                    setter = body._set_position
                    self._column_map.append(ColumnMapping(setter, axis_index, col))

            
            # Extract Rigid Body Marker informations
            elif asset_type == 'Rigid Body Marker':
                root = label.split(":")[0]
                if root in self.rigid_bodies:
                    if label in self.rigid_bodies[root].rigid_body_markers:
                        body = self.rigid_bodies[root].rigid_body_markers[label]
                    else:
                        body = RigidBodyMarker(label,ID)
                        self.rigid_bodies[root].rigid_body_markers[label] = body
                        
                    if field == 'Position':
                        axis_index = {'X':0, 'Y':1, 'Z':2}[axis]
                        setter = body._set_position
                        self._column_map.append(ColumnMapping(setter, axis_index, col))
                    
                    if field == 'Marker Quality':
                        axis_index = {'' : 0}[axis]
                        setter = body._set_position
                        self._column_map.append(ColumnMapping(setter, axis_index, col))
            
            
            # Extract marker informations
            elif asset_type == 'Marker':
                root = label.split(":")[0]
                if root in self.rigid_bodies:
                    if label in self.rigid_bodies[root].markers:
                        body = self.rigid_bodies[root].markers[label]
                    else:
                        body = Marker(label,ID)
                        self.rigid_bodies[root].markers[label] = body
                        
                    if field == 'Position':
                        axis_index = {'X':0, 'Y':1, 'Z':2}[axis]
                        setter = body._set_position
                        self._column_map.append(ColumnMapping(setter, axis_index, col))
                                    
            else:
                if label not in self._ignored_labels:
                    if verbose: print("Ignoring object %s of type %s." % (label, asset_type))
                    self._ignored_labels.add(label)
                    
        # the actual frame data begins with line 8, one frame per line, starting with frame 0
        return

    # ================================================================
    def _read_data(self, stream, verbose = False):
        """Process frame data rows from the CSV stream."""

        # Note that the frame_num indices do not necessarily start from zero,
        # but the setter functions assume that the array indices do.  This
        # implementation just ignores the original frame numbers, the frames are
        # renumbered from zero.
        for row_num, row in enumerate(stream):
            frame_num = int(row[0])
            frame_t   = float(row[1])
            values    = row[2:]

            # if verbose: print "Processing row_num %d, frame_num %d, time %f." % (row_num, frame_num, frame_t)

            # add the new frame time to each object storing a trajectory
            for body in self.rigid_bodies.values():
                body._add_frame(frame_t)
                for elem in body.rigid_body_markers.values():
                    elem._add_frame(frame_t)  
                for elem in body.markers.values():
                    elem._add_frame(frame_t)
            
            # add the new frame time to each object storing a trajectory
            for skeleton in self.skeletons.values():
                #skeleton._add_frame(frame_t)
                for elem in skeleton.bones.values():
                    elem._add_frame(frame_t)  
                for elem in skeleton.bone_markers.values():
                    elem._add_frame(frame_t)                               
            
            # process the columns of interest
            for mapping in self._column_map:
                # each mapping is a namedtuple with a setter method, column index, and axis name
                mapping.setter( row_num, mapping.axis, values[mapping.column] )

    # ================================================================
