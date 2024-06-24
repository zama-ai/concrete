"""
Detect sum sub-circuits.
"""

# Does not handle TLU-2 chains
#                 loss of chain
# ADD RUNNING SUMS AS A FALL BACK ON MISSING "CARRY"

# detect sum equivalence
# assuming carries has been found as representing a sum (no need to recheck)
# we only need to assert the results is linear for a combination new sources and we check
# this was done before so we could do similar


# approx by a brute force

# select the carry (carries are independant or same sum different bit_index), biggest first
# this define the sum we are extending
#    other are either new source or ignored if any dependancy among
#    collect applicable weights
#           new_sources are independant of carry
#           deduce new weights from carry bit index
#           args = new_sources_args + (sample carry args, saturate up to 2**16 total) (iterate first on hbs)

# bourrinage: only select independant args, gets the expanded sums, sample to check more on highest bits sum

# diff with before, new sources are indep with carry, ignore anything that relates to the carry
# no weight brute force, but select role

import itertools
import math
from dataclasses import dataclass
from enum import IntEnum
from functools import cached_property, lru_cache, total_ordering
from typing import Counter, Dict, List, Set, Tuple

from ..verilog_to_luts import TluNode, ValueNode


def is_power_of_2(n):
    return (n & (n - 1) == 0) and n != 0


def can_reduce_bit_width_weight(w):
    return w > 1 and is_power_of_2(w)


def remove_weights(t, w):
    return tuple(w_v for w_v in t if w != w_v[0])


def extract_bit(v, bit_index):
    return (v >> bit_index) % 2


def range_intersect_size(r1, r2):
    assert r1.step == r2.step == 1
    return len(range(max(r1.start, r2.start), min(r1.stop, r2.stop)))


@dataclass(frozen=True)
class Sum1b:
    bit_index: int
    bits: Tuple[Tuple[int, ValueNode]]
    positive: bool
    # dependencies: Set[ValueNode]

    @cached_property
    def max_value(self):
        return sum(w for w, _ in self.bits)

    @cached_property
    def min_non_zero(self):
        return min(w for w, _ in self.bits)

    @cached_property
    def max_bit_width(self):
        return int(math.ceil(math.log2(self.max_value + 1)))

    @cached_property
    def min_non_zero_bitwidth(self):
        return int(math.ceil(math.log2(self.min_non_zero + 1)))

    @cached_property
    def left_ignored_bits(self):
        return self.max_bit_width - (self.bit_index + 1)

    @cached_property
    def is_hsb(self):
        return self.left_ignored_bits == 0

    @cached_property
    def variable_bits_window(self):
        return range(self.min_non_zero_bitwidth - 1, self.max_bit_width)

    def eval(
        self,
        values: Dict[str, int],
        equivs: Dict[str, "Sum1b"],
        sum_values: Dict[str, int],
        pure_sum=False,
    ):
        added = 0
        for w, bit in self.bits:
            value = values.get(bit.name)
            if value is None:
                value = sum_values.get(bit.name)
            if value is None:
                equiv = equivs.get(bit.name)
                if equiv:
                    value = equiv.eval(values, equivs, sum_values)
                sum_values[bit.name] = value
            assert value is not None, bit.name
            added += w * value
        if pure_sum:
            return added
        bit = extract_bit(added, self.bit_index)
        assert bit in [0, 1]
        return bit if self.positive else (1 - bit)

    def deep_args(self, expand_up_to, equivs: Dict[str, "Sum1b"], leafs=None):
        leafs = set() if leafs is None else leafs
        for _, bit in self.bits:
            equiv = equivs.get(bit.name)
            if equiv and (equiv.dependencies & expand_up_to):
                equiv.deep_args(expand_up_to, equivs, leafs)
            else:
                leafs.add(bit.name)
        return leafs

    def shallow_expand(self, equivs: Dict[str, "Sum1b"]):
        has_expanded = False
        new_bit_index = self.bit_index
        # some bit equiv based on same sum should be grouped to be able to be expanded
        # because the bit extract is cancelled
        # e.g.
        # n16 = bit(0, (a[1] + b[1]))
        # n14 = bit(1, (a[1] + b[1]))
        # n16  + 2 * n14 == a[1] + b[1]
        grouped_sums_bits = {}
        cancelled_bit_extract = {}
        for weight, bit in self.bits:
            equiv = equivs.get(bit.name)
            if not equiv:
                continue
            key = (weight / 2**equiv.bit_index, equiv.sum_content)
            group = grouped_sums_bits.setdefault(key, set())
            group.add(equiv.bit_index)
            # TODO: relax based on weight and self.bit_index
            if len(group) >= equiv.max_bit_width:
                cancelled_bit_extract[key] = min(group)
        expansions = []
        equiv_last_significant = self.bit_index + 1
        significant_bits = set(range(self.min_non_zero_bitwidth - 1, equiv_last_significant))
        for weight, bit in self.bits:
            equiv = equivs.get(bit.name)
            if not equiv:
                expansions += [(weight, bit, None, True)]
                continue

            equiv_first_significant = equiv.min_non_zero_bitwidth - 1
            equiv_last_significant = equiv.max_bit_width - 1
            # take into account bit_index
            equiv_first_significant -= equiv.bit_index
            equiv_last_significant -= equiv.bit_index
            # weight should impact min and max, deepending on first bit and max weight
            equiv_last_significant = min(equiv_last_significant, equiv_last_significant)
            equiv_significant_bits = set(range(equiv_first_significant, equiv_last_significant + 1))
            intersect_size = len(significant_bits & equiv_significant_bits)
            key = (weight / 2**equiv.bit_index, equiv.sum_content)
            if (target_bit_index := cancelled_bit_extract.get(key)) is not None:
                if equiv.bit_index == target_bit_index:
                    has_expanded = True
                    expansions += [(weight, bit, equiv, True)]
            elif equiv.bit_index == 0 and weight == 2**self.bit_index:
                has_expanded = True
                expansions += [(weight, bit, equiv, True)]
            elif weight == 1 and intersect_size == 1:
                # the sum does not interact except for the equiv bit
                # we can only right expand once
                has_expanded = True
                new_bit_index = self.bit_index + equiv.bit_index
                expansions += [(weight, bit, equiv, False)]
                significant_bits = significant_bits | equiv_significant_bits
            else:
                if equiv:
                    pass
                expansions += [(weight, bit, None, True)]
        if not has_expanded:
            return None, []
        # expand
        new_bits = []
        rescale_factor = 2 ** (new_bit_index - self.bit_index)
        for weight, bit, expand_equiv, rescale in expansions:
            rescale_to = rescale_factor if rescale else 1
            if expand_equiv:
                for sub_weight, sub_bit in expand_equiv.bits:
                    new_weight = weight * sub_weight * rescale_to
                    new_bits.append((new_weight, sub_bit))
            else:
                new_weight = weight * rescale_to
                new_bits.append((new_weight, bit))
        new_self = Sum1b(
            bit_index=new_bit_index,
            bits=tuple(sorted(new_bits)),
            positive=object(),
            dependencies=self.dependencies,
        )
        expanded = [bit.name for _, bit, expanded, _ in expansions if expanded]
        # recurse
        new_new_self, new_expanded = new_self.shallow_expand(equivs)
        if new_new_self:
            return new_new_self, expanded + new_expanded
        return new_self, expanded

    def __str__(self):
        s = "" if self.positive else "1-"
        s += f"bit({self.bit_index}, "
        grouped_bits = {}
        for w, b in self.bits:
            grouped_bits.setdefault(w, []).append(b)
        for weight_i, (weight, bits) in enumerate(sorted(grouped_bits.items())):
            if weight_i > 0:
                s += " + "
            if weight > 1:
                s += f"{weight} * "
            bits = list(bits)
            if len(bits) == 1:
                s += f"{bits[0].name}"
            else:
                s += "("
                for bit_i, bit in enumerate(bits):
                    assert isinstance(bit, ValueNode), type(bit)
                    if bit_i > 0:
                        s += " + "
                    s += f"{bit.name}"
                s += ")"
        s += ")"
        return s

    def sum_str(self):
        s = ""
        grouped_bits = {}
        for w, b in self.bits:
            grouped_bits.setdefault(w, []).append(b)
        for weight_i, (weight, bits) in enumerate(sorted(grouped_bits.items())):
            if weight_i > 0:
                s += " + "
            if weight > 1:
                s += f"{weight} * "
            bits = list(bits)
            if len(bits) == 1:
                s += f"{bits[0].name}"
            else:
                s += "("
                for bit_i, bit in enumerate(bits):
                    assert isinstance(bit, ValueNode), type(bit)
                    if bit_i > 0:
                        s += " + "
                    s += f"{bit.name}"
                s += ")"
        return s

    @cached_property
    def sum_content(self):
        return tuple(sorted((w, b.name) for w, b in self.bits))


def tlu_get(tlu_node: TluNode, args: List[int]):
    content = tlu_node.content
    for arg in reversed(args):
        assert isinstance(content, list), content
        content = content[arg]
    assert isinstance(content, int), content
    return content


@lru_cache
def all_args_weight(bit_index, nb_args):
    assert nb_args
    domain_weight = [0] + [2**i for i in range(0, bit_index + 1)]
    all_weights = []
    for weights in itertools.product(domain_weight, repeat=nb_args):
        # needs at least 1 per bit and at least a start of 2 with weight 1
        if not (1 in weights):
            continue
        all_weights.append(weights)
    assert all_weights, (bit_index, nb_args)
    return sorted(all_weights, key=lambda v: sorted(v, reverse=True))


def interdependant_restricted_args_values(sum_equivs, args_equiv, tlu_node, dependencies_count):
    # Sometimes arguments of a TLU args are co-dependant, i.e. share some dependency
    # In that case shaking the full cross-domain of these args is not required.
    # Checking that the sum property holds on the full domain rejects valid sums.
    # This generates a subset of the full domain that is required to be checked and not more.
    # Eg. arg1 = bit(0, a + b)
    # Eg. arg2 = bit(1, a + b)
    # a b   -> arg1 arg2
    # 0 0         0    0
    # 0 1         1    0
    # 1 0         1    0
    # 1 1         0    1
    #       ->    1    1 never attained
    # As 1 1 is never attained for arg1 arg2, the TLU is free to associate any value for that case.
    # If checked we could reject a valid sum.
    # Note: this happens even for bare addition circuit when resticted to 2 bits TLU.
    interdependencies = []
    deep_args_groups = []
    for arg_equiv, arg in zip(args_equiv, tlu_node.arguments):
        deps = arg_equiv.dependencies if arg_equiv else []
        inter = {d for d in deps if dependencies_count[d] > 1}
        interdependencies.append(inter)
        if inter:
            deep_args_groups.append(arg_equiv.deep_args(inter, sum_equivs))
        else:
            deep_args_groups.append({arg.name})
    deep_args = {d for g in deep_args_groups for d in g}
    print(
        "Refine on deep args",
        deep_args,
        "for",
        ", ".join(a.name for a in tlu_node.arguments),
        "->",
        "".join(r.name for r in tlu_node.results),
    )
    deep_args_domain = list(itertools.product([0, 1], repeat=len(deep_args)))
    restricted_args_values = []
    # OPT: the deep_args detection could stop on solo shared co-nodes containing common dependencies
    for deep_args_value in deep_args_domain:
        deep_args_value_dict = {
            deep_arg: deep_arg_value for deep_arg, deep_arg_value in zip(deep_args, deep_args_value)
        }
        restricted_args_value = []
        for arg_equiv, arg in zip(args_equiv, tlu_node.arguments):
            if arg.name in deep_args_value_dict:
                arg_value = deep_args_value_dict[arg.name]
            else:
                assert arg_equiv
                arg_value = arg_equiv.eval(deep_args_value_dict, sum_equivs, dict())
            restricted_args_value.append(arg_value)
        restricted_args_values.append(tuple(restricted_args_value))
    restricted_args_values = set(restricted_args_values)
    return restricted_args_values


def check_tlu_sum_(sum_equivs: Dict[str, Sum1b], tlu_node: TluNode):
    args_equiv = [sum_equivs.get(arg.name) for arg in tlu_node.arguments]
    dependencies_count = Counter()
    for equiv, arg in zip(args_equiv, tlu_node.arguments):
        dependencies_count.update(equiv.dependencies if equiv else {arg: 1})
    dependencies = set(dependencies_count.keys())
    equiv = simple_check_tlu_sum(args_equiv, tlu_node, dependencies)
    if equiv:
        return equiv
    has_interdep = any(c > 1 for c in dependencies_count.values())
    if not has_interdep:
        print(
            f"No Detection (1) on ",
            ", ".join(a.name for a in tlu_node.arguments),
            "->",
            "".join(r.name for r in tlu_node.results),
        )
        return None
    restricted_args_values = interdependant_restricted_args_values(
        sum_equivs, args_equiv, tlu_node, dependencies_count
    )
    if len(restricted_args_values) < 2 ** len(tlu_node.arguments):
        equiv = simple_check_tlu_sum(args_equiv, tlu_node, dependencies, restricted_args_values)
        # arg_domain_restriction
        # expand dependencies until there is no longer an issue
        if equiv:
            element, *_ = restricted_args_values
            print("Refined detect", len(element))
            return equiv
    print(
        f"No Detection (2) on",
        ", ".join(a.name for a in tlu_node.arguments),
        "->",
        "".join(r.name for r in tlu_node.results),
    )
    return None


def simple_check_tlu_sum(
    args_equiv: List[Sum1b], tlu_node: TluNode, dependencies, args_domain=None
):
    nb_args = len(args_equiv)
    if nb_args == 0:
        return None
    args_domain = (
        list(itertools.product([0, 1], repeat=nb_args)) if args_domain is None else args_domain
    )
    highest_bit_index = nb_args - 1
    for bit_index in range(2 + highest_bit_index):
        for weights in all_args_weight(bit_index, nb_args):
            for positive in (True, False):
                for i, args in enumerate(args_domain):
                    weighted_sum = sum(
                        w * (v if not e or e.positive else (1 - v))
                        for w, v, e in zip(weights, args, args_equiv)
                    )
                    sum_bit = extract_bit(weighted_sum, bit_index)
                    tlu_result = tlu_get(tlu_node, args)
                    good = (sum_bit == tlu_result) == positive
                    if not good:
                        break
                else:
                    # print(f"Detection on ",  ",".join(sorted(a.name for a in tlu_node.arguments)), "->", "".join(r.name for r in tlu_node.results))
                    # print(f"\tas{' ' if positive else ' neg '}bit {bit_index}-th sum {weights}")
                    bits = list(sorted(zip(weights, tlu_node.arguments), key=lambda v: v[1].name))
                    return Sum1b(
                        bit_index=bit_index,
                        positive=positive,
                        bits=bits,
                        dependencies=dependencies,
                    )
    return None


class SumElment(IntEnum):
    # Ordered by preference
    Carry = 1
    NewElement = 2
    Ignored = 3

    @classmethod
    def any(cls):
        return (cls.Carry, cls.NewElement, cls.Ignored)

    @classmethod
    def no_carry(cls):
        return (cls.NewElement, cls.Ignored)


@lru_cache
def kinds_domain_cache(can_carry):
    domains = [SumElment.any() if b else SumElment.no_carry() for b in can_carry]
    cross_domain = itertools.product(*domains)
    # sort to search first on solution without ignore
    cross_domain = sorted(cross_domain, key=lambda v: sorted(v, reverse=True))
    return [
        kinds
        for kinds in cross_domain
        if sum(k != SumElment.Ignored for k in kinds) > 1
        # at least 2 elements must not be ignored
    ]


@lru_cache
def weights_domain_cache(kinds):
    domains = [(0,) if k == SumElment.Ignored else (1, 2) for k in kinds]
    cross_domain = itertools.product(*domains)
    # sort to search first on solution without ignore
    cross_domain = sorted(cross_domain, key=lambda v: sorted(v, reverse=True))
    return [
        kinds
        for kinds in cross_domain
        if sum(k != SumElment.Ignored for k in kinds) > 1
        # at least 2 elements must not be ignored
    ]


def check_tlu_sum(sum_equivs: List[Sum1b], tlu_node: TluNode):
    # censor carries that has no basis
    nb_args = len(tlu_node.arguments)
    if nb_args <= 1:
        return None
    args_equiv = tuple(sum_equivs.get(arg.name) for arg in tlu_node.arguments)
    can_carry = tuple(arg_equiv is not None for arg_equiv in args_equiv)
    kinds_domain = kinds_domain_cache(can_carry)
    if all(arg_equiv and arg_equiv.bit_index == 1 for arg_equiv in args_equiv):
        kinds_domain = kinds_domain_cache(tuple([False]) * len(can_carry))
    # expand/new/ignore, weight,
    args_values = [0] * len(args_equiv)
    for kinds in kinds_domain:
        check = False and tlu_node.name == "n16"
        if check:
            print("Kind", kinds)
            check = False and kinds == (SumElment.Carry, SumElment.Carry)
        weight_domains = weights_domain_cache(kinds)
        # collect all variable, new and from carry
        variables = []
        all_bit_index = []
        for arg, equiv, kind in zip(tlu_node.arguments, args_equiv, kinds):
            # SOME VARIABLES DOESN'T NEED TO BE IN DICT
            if kind == SumElment.Carry:
                all_bit_index.append(equiv.bit_index)
                # TODO: keeps higher weights only
                for _, bit in equiv.bits:
                    variables.append(bit.name)
            else:
                variables.append(arg.name)
        min_bit_index = max(all_bit_index, default=0)
        min_w = 2**min_bit_index
        variables_domain = list(itertools.product([0, 1], repeat=len(variables)))
        hyp_domain = itertools.product(
            weight_domains,
            range(min_bit_index, min_bit_index + max(2, math.floor(math.log2(2 * nb_args)))),
            (True, False),
        )
        for weights, bit_index, positive in hyp_domain:
            check = (
                False
                and tlu_node.name == "n16"
                and kinds == (SumElment.Carry, SumElment.Carry)
                and weights == (1, 1)
                and bit_index == 2
                and positive
            )
            if check:
                print("W", weights, bit_index, positive, "X", min_w)
            for variables_values in variables_domain:
                # if check:
                #     print("variables", variables, variables_values)
                # TODO: detect variables to update, do something faster for no carry variables
                values = dict(zip(variables, variables_values))
                weighted_sum = 0
                for i, (arg_equiv, arg, w) in enumerate(
                    zip(args_equiv, tlu_node.arguments, weights)
                ):
                    if arg.name in values:
                        args_values[i] = values[arg.name]
                        weighted_sum += values[arg.name] * min_w * w
                    else:
                        assert arg_equiv
                        # TODO: only do if something changed
                        pure_sum = arg_equiv.eval(values, sum_equivs, dict(), pure_sum=True)
                        args_values[i] = arg_equiv.eval(
                            values, sum_equivs, dict()
                        )  # TODO: remove that dict
                        # assert 0 <= args_values[i] <= 1
                        weighted_sum += pure_sum * w * 2 ** (min_bit_index - arg_equiv.bit_index)
                        if check:
                            print(
                                "CARRRY",
                                w * 2 ** (min_bit_index - arg_equiv.bit_index),
                                "*",
                                arg_equiv.sum_str(),
                                pure_sum * w * 2 ** (min_bit_index - arg_equiv.bit_index),
                            )
                tlu_result = tlu_get(tlu_node, args_values)
                sum_bit = extract_bit(weighted_sum, bit_index)
                good = (sum_bit == tlu_result) == positive
                if not good:
                    if check:
                        print(args_values, values)
                        print("Not good", weighted_sum, sum_bit, tlu_result)
                    break
            else:
                bits = []
                for arg, equiv, kind, weight in zip(tlu_node.arguments, args_equiv, kinds, weights):
                    if kind == SumElment.Carry:
                        for sub_weight, bit in equiv.bits:
                            bits.append(
                                (sub_weight * weight * 2 ** (min_bit_index - equiv.bit_index), bit)
                            )
                    elif kind == SumElment.NewElement:
                        bits.append((min_w * weight, arg))
                return Sum1b(
                    bit_index=bit_index,
                    positive=positive,
                    bits=list(sorted(bits, key=lambda v: v[1].name)),
                )
    return None


@dataclass(frozen=True)
class SumMb:
    bits_index: List[Tuple[int, ValueNode]]
    sum_bits: List[Tuple[int, ValueNode]]

    @property
    def bit_extract_uneeded(self):
        return all(
            b.base_name == self.bits_index[0][1].base_name and b.origin.bit_index == i
            for i, b in self.bits_index
        )

    def __str__(self):
        s = f"bits("
        if self.bit_extract_uneeded and not len(self.bits_index) == 1:
            s += f"{self.bits_index[0][1].base_name}[{min(i for i, _ in self.bits_index)}:{max(i for i, _ in self.bits_index)}], "
        else:
            results = ",".join(f"{v.name}:{index}" for index, v in self.bits_index)
            s += f"{results}, "
        grouped_bits = {}
        for w, b in self.sum_bits:
            grouped_bits.setdefault(w, []).append(b)
        for weight_i, (weight, bits) in enumerate(sorted(grouped_bits.items())):
            if weight_i > 0:
                s += " + "
            if weight > 1:
                s += f"{weight} * "
            bits = list(bits)
            if len(bits) == 1:
                s += f"{bits[0].name}"
            else:
                s += "("
                for bit_i, bit in enumerate(bits):
                    assert isinstance(bit, ValueNode), type(bit)
                    if bit_i > 0:
                        s += " + "
                    s += f"{bit.name}"
                s += ")"
        s += ")"
        return s


def detect(nodes: List[TluNode], _params):
    sum_equivs = {}
    sum_chaining = {}
    print("Local equiv")
    # find mono-tlu bit sum equivalence
    for node in nodes:
        new_equiv = check_tlu_sum(sum_equivs, node)
        if not new_equiv:
            print(node, flush=True)
            continue
        assert len(node.results) == 1
        sum_equivs[node.name] = new_equiv
        print(node.name, "=", new_equiv, "/", node, flush=True)

    return
    exit()
    print()
    used = set()
    unused_sum = set()
    for node in reversed(nodes):
        for r in node.results:
            if r.is_interface:
                used.add(r.name)
        equiv: Sum1b = sum_equivs.get(node.name)
        if equiv:
            has_1_used = False
            for r in node.results:
                if r.name in used:
                    has_1_used = True
                else:
                    unused_sum.add(r.name)
            if has_1_used:
                for _, bit in equiv.bits:
                    used.add(bit.name)
        else:
            for bit in node.arguments:
                used.add(bit.name)
            has_1_used = True

    # regroup compatible mono_bit sums
    # we simply start with bigger full sums
    sum_nodes = sorted(
        (node for node in nodes if node.name in sum_equivs and node.name in used),
        key=lambda n: len(sum_equivs[n.name].sum_content),
    )
    shared_full_sum = {}
    shared_full_sum_nodes_results = {}
    for node in reversed(sum_nodes):
        equiv: Sum1b = sum_equivs[node.name]
        shared_equiv_sum_content = shared_full_sum.get(equiv.sum_content, equiv.sum_content)
        assert len(node.results) == 1
        # TODO: error in equiv.bit_index
        shared_full_sum_nodes_results.setdefault(shared_equiv_sum_content, set()).add(
            (equiv.bit_index, node.results[0])
        )
        if shared_equiv_sum_content != equiv.sum_content:
            assert shared_equiv_sum_content in shared_full_sum_nodes_results
            continue
        shared_full_sum[shared_equiv_sum_content] = shared_equiv_sum_content
        smaller_equiv_sum_content = equiv.sum_content
        while can_reduce_bit_width_weight(smaller_equiv_sum_content[-1][0]):
            smaller_equiv_sum_content = remove_weights(
                smaller_equiv_sum_content, smaller_equiv_sum_content[-1][0]
            )
            if not smaller_equiv_sum_content:
                break
            # it's ok to overwrite
            shared_full_sum[smaller_equiv_sum_content] = shared_equiv_sum_content

    new_nodes = []
    done = set()
    for node in reversed(nodes):
        if node.name in unused_sum:
            continue
        equiv: Sum1b = sum_equivs.get(node.name)
        if not equiv:
            new_nodes.append(node)
            continue
        shared_equiv_sum_content = shared_full_sum[equiv.sum_content]
        results = shared_full_sum_nodes_results[shared_equiv_sum_content]
        used_bits_index = list(
            sorted(
                i_v
                for i_v in results
                if (i_v[1].name in used or i_v[1].is_interface) and not i_v[1].name in done
            )
        )
        for _, bit in used_bits_index:
            done.add(bit.name)
        if not used_bits_index:
            continue
        new_nodes.append(SumMb(used_bits_index, equiv.bits))

    new_nodes = reversed(new_nodes)
    print("PROG:\n")
    for n in new_nodes:
        print(str(n))


# Detection on  a[1],b[1] -> n16
# 	as bit 1-th sum (1, 1)
# Detection on  a[0],b[0] -> result[0]
# 	as bit 1-th sum (1, 1)
# Detection on  a[0],b[2] -> n17
# 	as bit 1-th sum (1, 1)
# Detection on  n16,result[0] -> n15
# 	as bit 1-th sum (1, 1)

# DEEP ARGS {'b[2]', 'n16', 'b[0]', 'a[0]'}
# NEW ARGS [(0, 0), (0, 1), (1, 0), (1, 1)]
# No Detection (2) on n15, n17 -> n14

# a0, b0, a1, b1,

# n17: a0, b2

# n42 =            bit(1, (a[1] + b[1]) + 2 * (n29 + n30)      ) / bit(0, (n28 + n29 + n30)) ['n28']
# result[2] =      bit(1, (a[1] + b[1]) + 2 * (n29 + n30 + n43)) / bit(0, (n42 + n43)) ['n42']
# bits(n42:1,result[2]:1, (a[1] + b[1]) + 2 * (n29 + n30 + n43))
