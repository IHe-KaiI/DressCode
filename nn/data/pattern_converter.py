import copy
from datetime import datetime
import numpy as np
from numpy.random import default_rng
from pathlib import Path
import sys
import torch

if sys.version_info[0] >= 3:
    from scipy.spatial.transform import Rotation as scipy_rot  # Not available in scipy 0.19.1 installed for Maya

# My modules
from pattern.core import panel_spec_template
from pattern.wrappers import VisPattern
from pattern import rotation as rotation_tools


# ------- Custom Errors --------
class EmptyPanelError(Exception):
    pass

class InvalidPatternDefError(Exception):
    """
        The given pattern definition (e.g. numeric representation) is not self-consistent.
        Examples: stitches refer to non-existing edges
    """
    def __init__(self, pattern_name='', message=''):
        self.message = 'Pattern {} is invalid'.format(pattern_name)
        if message:
            self.message += ': ' + message
        super().__init__(self.message)


# -------- Pattern Interface -----
class NNSewingPattern(VisPattern):
    """
        Interface to Sewing patterns with Neural Net friendly representation
    """
    def __init__(self, pattern_file=None, view_ids=False, panel_classifier=None, template_name=None):
        """
            `template_name` is need to use `panel_classifier` for panel ordering
        """
        self.panel_classifier = panel_classifier
        self.template_name = template_name

        super().__init__(pattern_file=pattern_file, view_ids=view_ids)        

    def pattern_as_tensors(
            self, 
            pad_panels_to_len=None, pad_panels_num=None, pad_stitches_num=None,
            with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Return pattern in format suitable for NN inputs/outputs
            * 3D tensor of panel edges
            * 3D tensor of panel's 3D translations
            * 3D tensor of panel's 3D rotations
        Parameters to control padding: 
            * pad_panels_to_len -- pad the list edges of every panel to this number of edges
            * pad_panels_num -- pad the list of panels of the pattern to this number of panels
        """
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::pattern_as_tensors() is only supported for Python 3.6+ and Scipy 1.2+')
        
        # get panel ordering
        panel_order = self.panel_order(pad_to_len=pad_panels_num)

        # Calculate max edge count among panels -- if not provided
        panel_lens = [len(self.pattern['panels'][name]['edges']) if name is not None else 0 for name in panel_order]
        max_len = pad_panels_to_len if pad_panels_to_len is not None else max(panel_lens)

        # Main info per panel
        panel_seqs, panel_translations, panel_rotations = [], [], []
        for panel_name in panel_order:
            if panel_name is not None:
                edges, rot, transl = self.panel_as_numeric(panel_name, pad_to_len=max_len)
            else:  # empty panel
                edges, rot, transl = self._empty_panel(max_len)
            panel_seqs.append(edges)
            panel_translations.append(transl)
            panel_rotations.append(rot)

        # Stitches info. Order of stitches doesn't matter
        stitches_num = len(self.pattern['stitches']) if pad_stitches_num is None else pad_stitches_num
        if stitches_num < len(self.pattern['stitches']):
            raise ValueError(
                'BasicPattern::Error::requested number of stitches {} is less the number of stitches {} in pattern {}'.format(
                    stitches_num, len(self.pattern['stitches']), self.name
                ))
        
        # Padded value is zero allows to treat the whole thing as index array
        # But need care when using -- as indexing will not crush when padded values are not filtered
        stitches_indicies = np.zeros((2, stitches_num), dtype=np.int) 
        if with_stitch_tags:
            # padding happens automatically, if panels are padded =)
            stitch_tags = self.stitches_as_tags()
            tags_per_edge = np.zeros((len(panel_seqs), len(panel_seqs[0]), stitch_tags.shape[-1]))
        for idx, stitch in enumerate(self.pattern['stitches']):
            for id_side, side in enumerate(stitch):
                panel_id = panel_order.index(side['panel'])
                edge_id = side['edge']
                stitches_indicies[id_side][idx] = panel_id * max_len + edge_id  # pattern-level edge id
                if with_stitch_tags:
                    tags_per_edge[panel_id][edge_id] = stitch_tags[idx]

        # format result as requested
        result = [np.stack(panel_seqs), np.array(panel_lens)]
        result.append(len(self.pattern['panels']))  # actual number of panels 
        if with_placement:
            result.append(np.stack(panel_rotations))
            result.append(np.stack(panel_translations))
        if with_stitches:
            result.append(stitches_indicies)
            result.append(len(self.pattern['stitches']))  # actual number of stitches
        if with_stitch_tags:
            result.append(tags_per_edge)

        return tuple(result) if len(result) > 1 else result[0]

    def pattern_from_tensors(
            self, pattern_representation, 
            panel_rotations=None, panel_translations=None, stitches=None,
            padded=False):
        """Create panels from given panel representation. 
            Assuming that representation uses cm as units"""
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::pattern_from_tensors() is only supported for Python 3.6+ and Scipy 1.2+')

        # Invalidate parameter & constraints values
        self._invalidate_all_values()

        # Assuming the input (from NN) follows the norm -- no updates will be made on further loads
        self.properties.update(
            curvature_coords='relative', 
            normalize_panel_translation=False, 
            normalized_edge_loops=True,
            units_in_meter=100  # cm
        )

        # remove existing panels -- start anew
        self.pattern['panels'] = {}
        in_panel_order = []
        new_panel_ids = [None] * len(pattern_representation)  # for correct stitches assignment in case of empty panels in-between
        for idx in range(len(pattern_representation)):
            panel_name = 'panel_' + str(idx) if self.panel_classifier is None else self.panel_classifier.class_name(idx)
            
            try:
                # if panel_rotations[idx].sum() == 0.:
                #     panel_rotations[idx][0] = 1.
                self.panel_from_numeric(
                    panel_name, 
                    pattern_representation[idx], 
                    rotation=panel_rotations[idx] if panel_rotations is not None else None,
                    translation=panel_translations[idx] if panel_translations is not None else None,
                    padded=padded)
                in_panel_order.append(panel_name)
                new_panel_ids[idx] = len(in_panel_order) - 1
            except EmptyPanelError as e:
                # Found an empty panel in the input -- moving on to the next one
                pass

        self.pattern['panel_order'] = in_panel_order  # save the incoming panel order

        # remove existing stitches -- start anew
        self.pattern['stitches'] = []
        if stitches is not None and len(stitches) > 0:
            if not padded:
                # TODO implement mapping of pattern-level edge ids -> (panel_id, edge_id) for panels with different number of edges
                raise NotImplementedError('BasicPattern::Recovering stitches for unpadded pattern is not supported')
            
            edges_per_panel = pattern_representation.shape[1]
            for stitch_id in range(stitches.shape[1]):
                stitch_object = []
                if stitches[0][stitch_id] == 0 and stitches[1][stitch_id] == 0:
                    # This is padding -- skip
                    continue 
                for side_id in range(stitches.shape[0]):
                    pattern_edge_id = stitches[side_id][stitch_id]
                    in_panel_id = int(pattern_edge_id // edges_per_panel)

                    if in_panel_id > (len(pattern_representation) - 1) or new_panel_ids[in_panel_id] is None:  # validity of stitch definition
                        raise InvalidPatternDefError(self.name, 'stitch {} referes to non-existing panel {}'.format(stitch_id, in_panel_id))
                    stitch_object.append(
                        {
                            "panel": in_panel_order[new_panel_ids[in_panel_id]],  # map to names of filteres non-empty panels
                            "edge": int(pattern_edge_id % edges_per_panel), 
                        }
                    )
                self.pattern['stitches'].append(stitch_object)
        else:
            print('BasicPattern::Warning::{}::Panels were updated but new stitches info was not provided. Stitches are removed.'.format(self.name))

    def panel_as_numeric(self, panel_name, pad_to_len=None):
        """
            Represent panel as sequence of edges with each edge as vector of fixed length plus the info on panel placement.
            * Edges are returned in additive manner: 
                each edge as a vector that needs to be added to previous edges to get a 2D coordinate of end vertex
            * Panel translation is represented with "universal" heuristic -- as translation of midpoint of the top-most bounding box edge
            * Panel rotation is returned as is but in quaternions

            NOTE: 
                The conversion uses the panels edges order as is, and 
                DOES NOT take resposibility to ensure the same traversal order of panel edges is used across datapoints of similar garment type.
                (the latter is done on sampling or on load)
        """
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::panel_as_numeric() is only supported for Python 3.6+ and Scipy 1.2+')

        panel = self.pattern['panels'][panel_name]
        vertices = np.array(panel['vertices'])
    
        # -- Construct the edge sequence in the recovered order --
        edge_sequence = [self._edge_as_vector(vertices, edge) for edge in panel['edges']]

        for i in range(len(edge_sequence)):
            edge_sequence[i][:2] = vertices[panel['edges'][i]['endpoints'][0]]
            
        # padding if requested
        if pad_to_len is not None:
            if len(edge_sequence) > pad_to_len:
                raise ValueError('BasicPattern::{}::panel {} cannot fit into requested length: {} edges to fit into {}'.format(
                    self.name, panel_name, len(edge_sequence), pad_to_len))
            for _ in range(len(edge_sequence), pad_to_len):
                edge_sequence.append(np.zeros_like(edge_sequence[0]))
        
        # ----- 3D placement convertion  ------
        # Global Translation (more-or-less stable across designs)
        translation, _ = self._panel_universal_transtation(panel_name)

        panel_rotation = scipy_rot.from_euler('xyz', panel['rotation'], degrees=True)  # pattern rotation follows the Maya convention: intrinsic xyz Euler Angles
        rotation_representation = np.array(panel_rotation.as_quat())

        return np.stack(edge_sequence, axis=0), rotation_representation, translation

    def panel_from_numeric(self, panel_name, edge_sequence, rotation=None, translation=None, padded=False):
        """ Updates or creates panel from NN-compatible numeric representation
            * Set panel vertex (local) positions & edge dictionaries from given edge sequence
            * Set panel 3D translation and orientation if given. Accepts 6-element rotation representation -- first two colomns of rotation matrix"""
        if sys.version_info[0] < 3:
            raise RuntimeError('BasicPattern::Error::panel_from_numeric() is only supported for Python 3.6+ and Scipy 1.2+')

        if padded:
            # edge sequence might be ending with pad values or the whole panel might be a mock object
            # print('[edge_sequence]', edge_sequence.shape)
            selection = ~np.all(np.isclose(edge_sequence, 0, atol=1.5), axis=1)  # only non-zero rows  
            # print('[selection]', selection.shape)   
            edge_sequence = edge_sequence[selection]
            if len(edge_sequence) < 3:
                # 0, 1, 2 edges are not enough to form a panel -> assuming this is a mock panel
                raise EmptyPanelError('{}::EmptyPanelError::Supplied <{}> is empty'.format(self.__class__.__name__, panel_name))

        if panel_name not in self.pattern['panels']:
            # add new panel! =)
            self.pattern['panels'][panel_name] = copy.deepcopy(panel_spec_template)

        # ---- Convert edge representation ----
        # vertices = np.array([[0, 0]])  # first vertex is always at origin
        vertices = np.zeros((0, 2))
        edges = []
        for idx in range(len(edge_sequence)):
            edge_info = edge_sequence[idx]
            vertices = np.vstack([vertices, edge_info[:2]])
            next_idx = idx + 1
            if next_idx >= len(edge_sequence): next_idx = 0
            edges.append(self._edge_dict(idx, next_idx, edge_info[2:4]))

        # last edge is a special case
        # idx = len(vertices) - 1
        # edge_info = edge_sequence[-1]
        # fin_vert = vertices[-1] + edge_info[:2]
        # if all(np.isclose(fin_vert, 0, atol=3)):  # 3 cm per coordinate is a tolerable error
        #     edges.append(self._edge_dict(idx, 0, edge_info[2:4]))
        # else:
        #     print('BasicPattern::Warning::{} with panel {}::Edge sequence do not return to origin. '
        #           'Creating extra vertex'.format(self.name, panel_name))
        #     vertices = np.vstack([vertices, fin_vert])
        #     edges.append(self._edge_dict(idx, idx + 1, edge_info[2:4]))

        # update panel itself
        panel = self.pattern['panels'][panel_name]
        panel['vertices'] = vertices.tolist()
        panel['edges'] = edges

        # ----- 3D placement setup --------
        if rotation is not None:
            rotation_obj = scipy_rot.from_quat(rotation)
            panel['rotation'] = rotation_obj.as_euler('xyz', degrees=True).tolist()

        if translation is not None:
            # we are getting translation of 3D top-midpoint (aka 'universal translation')
            # convert it to the translation from the origin 
            _, transl_origin = self._panel_universal_transtation(panel_name)

            shift = np.append(transl_origin, 0)  # to 3D
            panel_rotation = scipy_rot.from_euler('xyz', panel['rotation'], degrees=True)
            comenpensating_shift = - panel_rotation.as_matrix().dot(shift)
            translation = translation + comenpensating_shift

            panel['translation'] = translation.tolist()
        
    def stitches_as_tags(self, panel_order=None, pad_to_len=None):
        """For every stitch, assign an approximate identifier (tag) of the stitch to the edges that are part of that stitch
            * tags are calculated as ~3D locations of the stitch when the garment is draped on the body in T-pose
            * It's calculated as average of the participating edges' endpoint -- Although very approximate, this should be enough
            to separate stitches from each other and from free edges
        Return
            * List of stitch tags for every stitch in the panel
        """
        # NOTE stitch tags values are independent from the choice of origin & edge order within a panel
        # iterate over stitches
        stitch_tags = []
        for stitch in self.pattern['stitches']:
            edge_tags = np.empty((2, 3))  # two 3D tags per edge
            for side_idx, side in enumerate(stitch):
                panel = self.pattern['panels'][side['panel']]
                edge_endpoints = panel['edges'][side['edge']]['endpoints']
                # get 2D locations of participating vertices -- per panel
                edge_endpoints = np.array([
                    panel['vertices'][edge_endpoints[side]] for side in [0, 1]
                ])
                # Get edges midpoints (2D)
                edge_mean = edge_endpoints.mean(axis=0)

                # calculate their 3D locations
                edge_tags[side_idx] = self._point_in_3D(edge_mean, panel['rotation'], panel['translation'])

            # take average
            stitch_tags.append(edge_tags.mean(axis=0))

        return np.array(stitch_tags)

    def stitches_as_3D_pairs(self, stitch_pairs_num=None, non_stitch_pairs_num=None, randomize_edges=False, randomize_list_order=False):
        """
            Return a collection of edge pairs with each pair marked as stitched or not, with 
            edges represented as 3D vertex positions and (relative) curvature values.
            All stitched pairs that exist in the pattern are guaranteed to be included. 
            It's not guaranteed that the pairs would be unique (hence any number of pairs could be requested,
            regardless of the total number of unique pairs)

            * stitch_pairs -- number of edge pairs that are part of a stitch to return. Should be larger then the number of stitches.
            * non_stitch_pairs -- total number of non-connected edge pairs to return.
            * randomize_edges -- to randomize direction of edges and the order within each pair.
            * randomize_list_order -- to randomize the list of 
        """

        if stitch_pairs_num is not None and stitch_pairs_num < len(self.pattern['stitches']):
            raise ValueError(
                '{}::{}::Error::Requested less edge pairs ({}) that there are stitches ({})'.format(
                    self.__class__.__name__, self.name, stitch_pairs_num, len(self.pattern['stitches'])))

        rng = default_rng()  # new Numpy random number generator API

        # collect edges representations per panels
        edges_3d = self._3D_edges_per_panel(randomize_edges)

        # construct edge pairs (stitched & some random selection of non-stitched)
        pairs = []
        mask = []

        # ---- Stitched ----
        stitched_pairs_ids = set()
        # stitches
        for stitch in self.pattern['stitches']:
            pair = []
            try:
                for side in [0, 1]:
                    pair.append(edges_3d[stitch[side]['panel']][stitch[side]['edge']])
            except IndexError:
                # this might happen on incorrectly predicted panels
                print(f'Warning::{self.name}::Missing edge while constructing stitch pairs')
                continue

            if randomize_edges and rng.integers(2):  # randomly change the order in pair
                # flip the edge
                pair[0], pair[1] = pair[1], pair[0]

            pairs.append(np.concatenate(pair))
            mask.append(True)
            stitched_pairs_ids.add((
                (stitch[0]['panel'], stitch[0]['edge']),
                (stitch[1]['panel'], stitch[1]['edge'])
            ))
        if stitch_pairs_num is not None and stitch_pairs_num > len(stitched_pairs_ids):
            for _ in range(len(stitched_pairs_ids), stitch_pairs_num):
                # choose of the existing pairs to duplicate
                pairs.append(pairs[rng.integers(len(stitched_pairs_ids))])
                mask.append(True)
        
        if non_stitch_pairs_num is not None:
            panel_order = self.panel_order()
            if len(pairs) < stitch_pairs_num:
                # e.g., no stitches constructed at all
                non_stitch_pairs_num += stitch_pairs_num - len(pairs)
            for _ in range(non_stitch_pairs_num):
                while True:
                    # random pairs
                    pair_names, pair_edges = [], []
                    for _ in [0, 1]:
                        pair_names.append(panel_order[rng.integers(len(panel_order))])
                        pair_edges.append(rng.integers(len(self.pattern['panels'][pair_names[-1]]['edges'])))

                    if pair_names[0] == pair_names[1] and pair_edges[0] == pair_edges[1]:
                        continue  # try again

                    # check if pair is already used
                    pair_id = ((pair_names[0], pair_edges[0]), (pair_names[1], pair_edges[1]))
                    if pair_id in stitched_pairs_ids or (pair_id[1], pair_id[0]) in stitched_pairs_ids:
                        continue  # try again -- accudentially came up with a stitch

                    # success! Use it
                    pairs.append(np.concatenate([edges_3d[pair_names[0]][pair_edges[0]], edges_3d[pair_names[1]][pair_edges[1]]]))
                    mask.append(False)  # at this point, all pairs are non-stitched!

                    break 
            
        if randomize_list_order:
            permutation = rng.permutation(len(pairs))
            return np.stack(pairs)[permutation], np.array(mask, dtype=bool)[permutation]
        else:
            return np.stack(pairs), np.array(mask, dtype=bool)

    def stitches_from_pair_classifier(self, model, data_stats):
        """ Update stitches in the pattern by predictions of edge pairs classification model"""

        self.pattern['stitches'] = []
        model.eval()

        edge_pairs_list, pairs_mapping, _ = self.all_edge_pairs(device=model.device_ids[0])
        print('[edge pairs list]', edge_pairs_list.shape)

        # apply appropriate scaling
        shift = torch.tensor(data_stats['f_shift'], device=model.device_ids[0])
        scale = torch.tensor(data_stats['f_scale'], device=model.device_ids[0])
        edge_pairs_list = (edge_pairs_list - shift) / scale

        preds = model(edge_pairs_list)
        preds_probability = torch.sigmoid(preds)
        preds_class = torch.round(preds_probability)

        # record stitches
        stitched_ids = preds_class.nonzero(as_tuple=False).squeeze().cpu().tolist()
        if len(preds_class.nonzero(as_tuple=False).squeeze().cpu().shape) == 0:
            stitched_ids = [stitched_ids]
        if len(stitched_ids) > 0:  # some stitches found!
            for stitch_idx in range(len(stitched_ids)):
                edge_pair = pairs_mapping[stitch_idx]

                self.pattern['stitches'].append(self._stitch_entry(
                    edge_pair[0][0], edge_pair[0][1], 
                    edge_pair[1][0], edge_pair[1][1], 
                    score=preds[stitched_ids[stitch_idx]].cpu().tolist()
                ))

        # Post-analysis: check if any of the edges parttake in multiple stitches & only leave the stronger ones
        to_remove = set()
        for base_stitch_id in range(len(self.pattern['stitches'])):
            base_stitch = self.pattern['stitches'][base_stitch_id]
            for side in [0, 1]:
                base_edge = base_stitch[side]
                for other_stitch_id in range(base_stitch_id + 1, len(self.pattern['stitches'])):
                    curr_stitch = self.pattern['stitches'][other_stitch_id]
                    if (base_edge['panel'] == curr_stitch[0]['panel'] and base_edge['edge'] == curr_stitch[0]['edge']
                            or base_edge['panel'] == curr_stitch[1]['panel'] and base_edge['edge'] == curr_stitch[1]['edge']):
                        # same edge, multiple stitches!
                        # score is the same for both sides, so it doesn't matter which one to take
                        to_remove.add(
                            base_stitch_id if base_stitch[0]['score'] < curr_stitch[0]['score'] else other_stitch_id)

        if len(to_remove):
            self.pattern['stitches'] = [value for i, value in enumerate(self.pattern['stitches']) if i not in to_remove]

    def all_edge_pairs(self, device='cpu'):
        """
            Construct all possible edge pairs for given sewing pattern 
            with GT stitching labels if available and requested in `with_labels`
        """
        edges_3D = self._3D_edges_per_panel()
        num_panels = len(self.panel_order())

        stitch_set = self._stitches_as_set()
        mask = []

        edge_pairs_list = []
        pairs_mapping = []
        for i in range(num_panels):
            panel_i = self.panel_order()[i]
            edges_i = np.array(edges_3D[panel_i])
            for j in range(i + 1, num_panels):  # assuming panels are not connected to themselves
                panel_j = self.panel_order()[j]
                edges_j = np.array(edges_3D[panel_j])

                rows, cols = np.indices((len(edges_i), len(edges_j)))
                edge_pairs = np.concatenate([edges_i[rows], edges_j[cols]], axis=-1)

                # record the pair
                edge_pairs = torch.from_numpy(edge_pairs).float().to(device)
                edge_pairs = edge_pairs.view(-1, edge_pairs.shape[-1])  # flatten to the list of pairs
                edge_pairs_list.append(edge_pairs)

                # record backward mapping & labels
                for row_idx in range(len(edges_i)):
                    for col_idx in range(len(edges_j)):
                        pair_id = ((panel_i, row_idx), (panel_j, col_idx))
                        pairs_mapping.append(pair_id)
                        
                        mask.append(pair_id in stitch_set or (pair_id[1], pair_id[0]) in stitch_set)

        if len(edge_pairs_list) == 0:
            raise InvalidPatternDefError(self.name, 'No edges to construct')

        edge_pairs_list = torch.cat(edge_pairs_list)

        return edge_pairs_list, pairs_mapping, mask

    def _stitches_as_set(self):
        stitches_set = set()
        for stitch in self.pattern['stitches']:
            stitches_set.add((
                (stitch[0]['panel'], stitch[0]['edge']),
                (stitch[1]['panel'], stitch[1]['edge'])
            ))
        return stitches_set

    def _edge_dict(self, vstart, vend, curvature):
        """Convert given info into the proper edge dictionary representation"""
        edge_dict = {'endpoints': [vstart, vend]}
        if not all(np.isclose(curvature, 0, atol=0.01)):  # 0.01 is tolerable error for local curvature coords
            edge_dict['curvature'] = curvature.tolist()
        return edge_dict

    def _3D_edges_per_panel(self, randomize_direction=False):
        """ 
            Return all edges in the pattern (grouped by panels)
            represented as 3D vertex positions and (relative) curvature values.

            * 'randomize_direction' -- request to randomly flip the direction of some edges
        """
        if randomize_direction:
            rng = default_rng()  # new Numpy random number generator API

        # collect edges representations per panels
        edges_3d = {}
        for panel_name in self.panel_order():
            edges_3d[panel_name] = []
            panel = self.pattern['panels'][panel_name]
            vertices = np.array(panel['vertices'])

            # To 3D
            rot_matrix = rotation_tools.euler_xyz_to_R(panel['rotation'])
            vertices_3d = np.stack([self._point_in_3D(vertices[i], rot_matrix, panel['translation']) for i in range(len(vertices))])

            # edge feature
            for edge_dict in panel['edges']:
                edge_verts = vertices_3d[edge_dict['endpoints']]  # ravel does not copy elements
                curvature = np.array(edge_dict['curvature']) if 'curvature' in edge_dict else [0, 0]

                if randomize_direction and rng.integers(2):
                    # flip the edge
                    edge_verts[[0, 1], :] = edge_verts[[1, 0], :]

                    curvature[0] = 1 - curvature[0] if curvature[0] else 0
                    curvature[1] = -curvature[1] 

                edges_3d[panel_name].append(np.concatenate([edge_verts.ravel(), curvature]))

        return edges_3d

    def _stitch_entry(self, panel_1, edge_1, panel_2, edge_2, score=None):
        """ element of a stitch list with given parameters (all need to be json-serializible)"""
        return [
            {
                'panel': panel_1, 
                'edge': edge_1, 
                'score': score
            },
            {
                'panel': panel_2, 
                'edge': edge_2, 
                'score': score
            },
        ]

    def _empty_panel(self, max_edge_num):
        """ Shape, rotation, and translation for empty panels"""
        # edge is 4-elem vector, 4 rotation element for quaternion, 3 element for world translation
        return np.zeros((max_edge_num, 4)), np.zeros(4), np.zeros(3)

    # ordering of panels according to classification
    def panel_order(self, force_update=False, pad_to_len=None):
        """
            Return order of panels either 
                * according to the one provided in the pattern spec
                * According to external panels classification if self.panel_classifier is set!
            Note: 'None' represent empty panels at that place of ordered elements

            Reloading 'panel_order' instead of 'define_panel_order' to preserve order from file 
                if self.panel_classifier is not defined and 'force_update' is false
        """
        self.panel_classifier = None # TODO: force update the panel order
        if self.panel_classifier is None or self.template_name is None:
            # preserves the order is given in pattern spec!
            order = super().panel_order(force_update=force_update)
        else:  
            # NOTE: re-evaluate even if `force_update` flag is false
            # as we need update even if the pattern spec already contains some order

            # construct the order according to class indices
            # -None- represents empty panels-placeholders
            order = [None] * len(self.panel_classifier)
            for panel_name in self.pattern['panels']:
                class_idx = self.panel_classifier.class_idx(self.template_name, panel_name)
                order[class_idx] = panel_name
        
        # Additionally pad to requested value if given
        if pad_to_len is not None:
            if pad_to_len < len(order):
                raise ValueError(
                    f'{self.__class__.__name__}::{self.name}::Error::Requested max num of panels {pad_to_len} '
                    f'is smaller then evaluated number of panels {len(order)}')
            order += [None] * (pad_to_len - len(order))

        # Remember the order for future reference
        self.pattern['panel_order'] = order

        return order


# ---------- test -------------
if __name__ == "__main__":
    # the pattern converter loading check
    import sys
    sys.path.append('./nn')
    from pathlib import Path
    from datetime import datetime
    import customconfig
    from pattern.wrappers import VisPattern
    from data.panel_classes import PanelClasses

    np.set_printoptions(precision=4, suppress=True)

    system_config = customconfig.Properties('./system.json')
    base_path = system_config['output']

    # NOTE 
    pattern = NNSewingPattern(
        Path(system_config['datasets_path']) / 'jacket_2200' / 'jacket_template_specification.json', 
        panel_classifier=PanelClasses('./nn/data_configs/panel_classes_condenced.json'), 
        template_name='jacket')

    empty_pattern = NNSewingPattern(panel_classifier=PanelClasses('./nn/data_configs/panel_classes_condenced.json'))
    print(pattern.panel_order())

    tensor, edge_lens, num_panels, rot, transl, stitches, stitch_num, stitch_tags = pattern.pattern_as_tensors(
        with_placement=True, with_stitches=True, with_stitch_tags=True)

    print(tensor.shape, edge_lens, num_panels, rot.shape, transl.shape, stitches.shape, stitch_num, stitch_tags.shape)

    print(tensor.max(), tensor.min(), transl.max(), transl.min())
    C = 1000.
    tensor = (tensor * C).astype(np.int32) / C
    rot = (rot * C).astype(np.int32) / C
    transl = (transl * C).astype(np.int32) / C
    stitch_tags = (stitch_tags * C).astype(np.int32) / C
    print(stitch_tags)
    empty_pattern.pattern_from_tensors(tensor, rot, transl, stitches, padded=True)
    print(empty_pattern.pattern['stitches'])

    # Save
    empty_pattern.name = pattern.name + 'from_empty_with_class' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S')
    pattern.name = pattern.name + '_with_class' + '_' + datetime.now().strftime('%y%m%d-%H-%M-%S')
    
    empty_pattern.serialize(system_config['output'], to_subfolder=True)
    pattern.serialize(system_config['output'], to_subfolder=True)
