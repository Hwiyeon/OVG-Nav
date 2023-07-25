import numpy as np

# 21 categories
obj_names = ["chair",             # 0
             "table",             # 1
             "picture",           # 2
             "cabinet",           # 3
             "cushion",           # 4
             "sofa",              # 5
             "bed",               # 6
             "chest_of_drawers",  # 7
             "plant",             # 8
             "sink",              # 9
             "toilet",            # 10
             "stool",             # 11
             "towel",             # 12
             "tv_monitor",        # 13
             "shower",            # 14
             "bathhub",           # 15
             "counter",           # 16
             "fireplace",         # 17
             "seating",           # 18
             "gym_equipment",     # 19
             "clothes"            # 20
             ]

## rednet 21 categories
rednet_obj_names = ["chair",             # 0
             "table",             # 1
             "picture",           # 2
             "cabinet",           # 3
             "cushion",           # 4
             "sofa",              # 5
             "bed",               # 6
             "chest_of_drawers",  # 7
             "plant",             # 8
             "sink",              # 9
             "toilet",            # 10
             "stool",             # 11
             "towel",             # 12
             "tv_monitor",        # 13
             "shower",            # 14
             "bathtub",           # 15
             "counter",           # 16
             "fireplace",         # 17
             "gym_equipment",     # 18
             "seating",           # 19
             "clothes"            # 20
             ]
category_to_task_category_id = {
    "chair": 0,
    "table": 1,
    "picture": 2,
    "cabinet": 3,
    "cushion": 4,
    "sofa": 5,
    "bed": 6,
    "chest_of_drawers": 7,
    "plant": 8,
    "sink": 9,
    "toilet": 10,
    "stool": 11,
    "towel": 12,
    "tv_monitor": 13,
    "shower": 14,
    "bathtub": 15,
    "counter": 16,
    "fireplace": 17,
    "gym_equipment": 18,
    "seating": 19,
    "clothes": 20,
}

# detection categories

obj_names_det = ['chair',         # 0
                 'sofa',         # 1
                 'plant',  # 2
                 'bed',           # 3
                 'toilet',        # 5
                 'tv_monitor',            # 6
                 'dining table',  # 4
                 'laptop',        # 7
                 'microwave',     # 8
                 'oven',          # 9
                 'sink',          # 10
                 'refrigerator',  # 11
                 'clock',         # 12
                 'vase',          # 13
                 ]



# 6 goal categories
gibson_goal_obj_names = ['chair',         # 0
                  'couch',         # 1
                  'potted plant',  # 2
                  'bed',           # 3
                  'toilet',        # 4
                  'tv'             # 5
                  ]

mp3d_goal_obj_names = ['chair',         # 0
                  'sofa',         # 1
                  'plant',  # 2
                  'bed',           # 3
                  'toilet',        # 4
                  'tv_monitor'             # 5
                  ]


# # room categories
# room_names = ['livingroom',    # 0
#               'bedroom',       # 1
#               'kitchen',       # 2
#               'bathroom',      # 3
#               'balcony',       # 4
#               'laundryroom',   # 5
#               'hallway',       # 6
#               'office',        # 7
#               'outdoor',       # 8
#               'others'         # 9
# ]
#
# def assign_room_category(category, name=False):
#     """
#     category : detailed room category
#     name : return the str name
#     out : broad room category idx of room_names
#     """
#
#     room_category_idx = 9  ## default is other
#     if category in ['livingroom','familyroom/lounge', 'lounge', 'tv']: # livingroom
#         room_category_idx = 0
#     elif category in ['bedroom']: # bedroom
#         room_category_idx = 1
#     elif category in ['kitchen', 'dining room', 'bar', 'dining booth']: # kitchen
#         room_category_idx = 2
#     elif category in ['bathroom', 'spa/sauna', 'toilet']: # bathroom
#         room_category_idx = 3
#     elif category in ['balcony', 'porch/terrace/deck']: # balcony
#         room_category_idx = 4
#     elif category in ['laundryroom', 'laundryroom/mudroom', 'closet']: # laundryroom
#         room_category_idx = 5
#     elif category in ['hallway', 'stairs', 'entryway/foyer/lobby']: # hallway
#         room_category_idx = 6
#     elif category in ['office', 'classroom', 'meetingroom/conferenceroom', 'library']: # office
#         room_category_idx = 7
#     elif category in ['outdoor']: # outdoor
#         room_category_idx = 8
#     elif category in ['other room', 'rec/game', 'workout/gym/exercise', 'utilityroom/toolroom', 'junk', 'garage']: # others
#         room_category_idx = 9
#
#     if name:
#         return room_names[room_category_idx]
#     else:
#         return room_category_idx


# simplified room categories
room_names = ['livingroom',  # 0
              'bedroom',  # 1
              'kitchen',  # 2
              'diningroom',  # 3
              'bathroom',  # 4
              'office',  # 5
              'hallway',  # 6
              'others'  # 7
              ]

mp3d_room_names = [
    'livingroom','familyroom', 'lounge',
    'tv', 'bedroom', 'kitchen',
    'dining room', 'bar', 'dining booth',
    'bathroom', 'spa', 'sauna',
    'toilet', 'balcony', 'porch', 'terrace', 'deck',
    'laundryroom', 'mudroom', 'closet', 'hallway',
    'stairs', 'entryway', 'lobby',
    'office', 'classroom', 'meetingroom', 'conferenceroom',
    'library', 'outdoor', 'gym', 'utilityroom' ,'toolroom', 'junk', 'garage'
]



def assign_room_category(category, name=False):
    """
    category : detailed room category
    name : return the str name
    out : broad room category idx of room_names
    """

    room_category_idx = 7  ## default is other
    if category in ['livingroom', 'familyroom/lounge', 'lounge', 'tv']:  # livingroom
        room_category_idx = 0
    elif category in ['bedroom']:  # bedroom
        room_category_idx = 1
    elif category in ['kitchen']:  # kitchen
        room_category_idx = 2
    elif category in ['dining room', 'dining booth']:  # diningroom
        room_category_idx = 3
    elif category in ['bathroom', 'toilet']:  # bathroom
        room_category_idx = 4
    elif category in ['office', 'classroom', 'meetingroom/conferenceroom', 'library']:  # office
        room_category_idx = 5
    elif category in ['hallway', 'stairs', 'entryway/foyer/lobby']:  # hallway
        room_category_idx = 6
    elif category in ['other room', 'rec/game', 'workout/gym/exercise', 'utilityroom/toolroom', 'junk',
                      'garage', 'outdoor', 'laundryroom', 'laundryroom/mudroom', 'closet',
                      'balcony', 'porch/terrace/deck', 'bar', 'spa/sauna']:  # others
        room_category_idx = 7

    if name:
        return room_names[room_category_idx]
    else:
        return room_category_idx

d3_40_colors_rgb = np.array(
    [
        [31, 119, 180],
        [174, 199, 232],
        [255, 127, 14],
        [255, 187, 120],
        [44, 160, 44],
        [152, 223, 138],
        [206, 219, 156],
        [140, 109, 49],
        [148, 103, 189],
        [197, 176, 213],
        [140, 86, 75],
        [196, 156, 148],
        [227, 119, 194],
        [247, 182, 210],
        [127, 127, 127],
        [199, 199, 199],
        [188, 189, 34],
        [219, 219, 141],
        [23, 190, 207],
        [158, 218, 229],
        [57, 59, 121],
        [82, 84, 163],
        [107, 110, 207],
        [156, 158, 222],
        [99, 121, 57],
        [140, 162, 82],
        [181, 207, 107],
        [214, 39, 40],
        [255, 152, 150],
        [189, 158, 57],
        [231, 186, 82],
        [231, 203, 148],
        [132, 60, 57],
        [173, 73, 74],
        [214, 97, 107],
        [231, 150, 156],
        [123, 65, 115],
        [165, 81, 148],
        [206, 109, 189],
        [222, 158, 214],
    ],
    dtype=np.uint8,
)