import os
import hashlib
import base64
import pprint
import boto3
from time import time
from bigtree import list_to_tree


# template to generate an html index
html_a_format = '<a href="{}">{}</a><br/>\n'
html_index_format = """
<!DOCTYPE html>
<html>
  <body>
    {}
  </body>
</html>
"""


def get_s3_bucket_by_name(name: str):
    """Get the s3 bucket with the given name.

    The function assumes there is a bucket with the given name, and will fail otherwise.

    Args:
        name (str): bucket name

    Returns:
        s3 bucket with the given name
    """
    s3 = boto3.resource("s3")
    buckets = s3.buckets.all()
    filtered = list(filter(lambda b: b.name == name, buckets))
    assert len(filtered) == 1, f"Expected a single bucket, but found {len(filtered)}"
    return filtered[0]


def objects_to_file_tree(objects):
    """Get the file tree given objects in an s3 bucket.

    It assumes object keys represent directories (e.g. concrete-python/* are all under concrete-python directory).

    Args:
        objects (list of s3 objects): objects from an s3 bucket

    Returns:
        file tree of the given s3 objects
    """
    paths = []
    for obj in objects:
        # we prefix all objects with 'root/' so that we get a file tree that is all under a unique directory.
        # this is considered later on and should be removed to compute object keys
        paths.append(f"root/{obj.key}")
    return list_to_tree(paths)


def build_indexes(file_tree):
    """Build an html index for every directory in the file tree.

    Args:
        file_tree: the file tree we build indexes for

    Returns:
        dict: html index per object key (e.g. {"concrete-python/index.html": "HTML INDEX"})
    """
    index_per_path = {}
    files = []
    for child in file_tree.children:
        # if it's a direcoty then we call the function recursively to build indexes of that directory
        if not child.is_leaf:
            child_index_per_path = build_indexes(child)
            index_per_path.update(child_index_per_path)
            # we build a link relative to the current directory
            link = f"{child.name}/index.html"
            files.append((link, child.name))
        # if it's a file then we add it to the list of files of the current directory to index it
        else:
            # we don't need to index index files
            if child.name == "index.html":
                continue
            # remove "/root" and build link from root '/'
            assert child.path_name.startswith("/root")
            link = child.path_name.removeprefix("/root")
            files.append((link, child.name))

    # remove "/root" and append the index filename
    if file_tree.is_root:
        index_path = "index.html"
    else:
        assert file_tree.path_name.startswith("/root/")
        index_path = file_tree.path_name.removeprefix("/root/") + "/index.html"

    # Build the html index of the current directory
    refs = ""
    for f in files:
        html_a = html_a_format.format(f[0], f[1])
        refs = refs + html_a
    index_per_path[index_path] = html_index_format.format(refs)
    return index_per_path


def invalidate_cloudfront_cache(distribution_id, items_to_invalidate):
    """Invalidate CloudFront cache for a list of items.

    Args:
        distribution_id (str): CloudFront distribution id
        items_to_invalidate (List[str]): list of items to invalidate

    Returns:
        dict: invalidation response
    """
    client = boto3.client("cloudfront")
    return client.create_invalidation(
        DistributionId=distribution_id,
        InvalidationBatch={
            "Paths": {
                "Quantity": len(items_to_invalidate),
                "Items": items_to_invalidate,
            },
            "CallerReference": str(time()).replace(".", ""),
        },
    )


if __name__ == "__main__":
    # retrieve bucket
    s3_bucket_name = os.environ.get("S3_BUCKET_NAME")
    if s3_bucket_name is None:
        raise RuntimeError("S3_BUCKET_NAME env variable should be set")
    bucket = get_s3_bucket_by_name(s3_bucket_name)
    # get all objects in the bucket
    objects = list(bucket.objects.all())
    # build a file_tree from the list of objects
    file_tree = objects_to_file_tree(objects)
    # build html indexes for every directory in the file_tree
    index_per_path = build_indexes(file_tree)
    # upload indexes to the appropriate location
    for path, index in index_per_path.items():
        # we log html indexes and their key (location) in the bucket
        print(f"Writing index to {path}:\n{index}\n")
        # body has to be bytes
        body = index.encode()
        # checksum is the base64 encoded md5
        body_checksum = base64.b64encode(hashlib.md5(body).digest()).decode()
        # ContentType isn't inferred automatically, so we specify it to make sure the server
        # handles it properly (don't download a file when requested)
        bucket.put_object(
            Key=path, Body=body, ContentMD5=body_checksum, ContentType="text/html"
        )
    # invalidate cache for the indexes
    cloudfront_distribution_id = os.environ.get("CLOUDFRONT_DISTRIBUTION_ID")
    if cloudfront_distribution_id is None:
        raise RuntimeError("CLOUDFRONT_DISTRIBUTION_ID env variable should be set")
    keys_to_invalidate = ["/" + k for k in index_per_path.keys()]
    print("Invalidating CloudFront cache for the following keys:", keys_to_invalidate)
    response = invalidate_cloudfront_cache(
        cloudfront_distribution_id, keys_to_invalidate
    )
    print("CloudFront invalidation response:")
    pprint.pprint(response)
