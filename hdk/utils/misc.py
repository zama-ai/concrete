"""Misc. utils for hdk"""


def get_unique_id():
    """Function to get a unique ID"""

    if not hasattr(get_unique_id, "generator"):

        def generator():
            current_id = 0
            while True:
                yield current_id
                current_id += 1

        get_unique_id.generator = generator()

    return next(get_unique_id.generator)
