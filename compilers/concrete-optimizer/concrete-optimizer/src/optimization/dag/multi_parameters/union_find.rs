pub struct UnionFind {
    pub parent: Vec<usize>,
}

impl UnionFind {
    // Used to detect instructions connected in levelled block (in partionning.rs).
    pub fn new(size: usize) -> Self {
        Self {
            parent: (0..size).collect(),
        }
    }

    pub fn find_canonical(&mut self, a: usize) -> usize {
        let parent = self.parent[a];
        if a == parent {
            return a;
        }
        let canonical = self.find_canonical(parent);
        self.parent[a] = canonical;
        canonical
    }

    pub fn union(&mut self, a: usize, b: usize) {
        _ = self.united_common_ancestor(a, b);
    }

    // use slow path compression, immediate parent check and early recognition
    pub fn united_common_ancestor(&mut self, a: usize, b: usize) -> usize {
        let parent_a = self.parent[a];
        let parent_b = self.parent[b];
        if parent_a == parent_b {
            return parent_a; // common ancestor
        }
        let common_ancestor = if a == parent_a && parent_b < parent_a {
            // uniting class_a the smallest b ancestor
            parent_b
        } else if b == parent_b && parent_a < parent_b {
            // uniting class_b the smallest b ancestor
            parent_a
        } else {
            self.united_common_ancestor(parent_a, parent_b) // loop
        };
        // classic path compression
        self.parent[a] = common_ancestor;
        self.parent[b] = common_ancestor;
        common_ancestor
    }
}

#[cfg(test)]
mod tests {

    use super::super::partitionning::Blocks;
    use super::*;

    #[test]
    fn test_union_find() {
        let size = 10;
        let mut uf = UnionFind::new(size);
        for i in 0..size {
            assert!(uf.find_canonical(0) == 0);
            assert!(uf.find_canonical(i) == i);
            uf.union(i, 0);
            assert!(uf.find_canonical(i) == 0, "{} {:?}", i, &uf.parent[0..=i]);
        }
        eprintln!("{:?}", uf.parent);
        let expected_group: Vec<usize> = (0..10).collect();
        assert!(Blocks::from(uf).blocks == vec![expected_group]);
    }
}
