# -*- coding: utf-8 -*-
from sage.all import round, log, oo
from dataclasses import dataclass


@dataclass
class Cost:
    """
    Algorithms costs.
    """

    rop: float = oo
    tag: str = None

    # An entry is "impermanent" if it grows when we run the algorithm again. For example, `δ`
    # would not scale with the number of operations but `rop` would. This check is strict such that
    # unknown entries raise an error. This is to enforce a decision on whether an entry should be
    # scaled.

    impermanents = {
        "rop": True,
        "repetitions": False,
        "tag": False,
        "problem": False,
    }

    @classmethod
    def register_impermanent(cls, data=None, **kwds):
        if data is not None:
            for k, v in data.items():
                if cls.impermanents.get(k, v) != v:
                    raise ValueError(f"Attempting to overwrite {k}:{cls.impermanents[k]} with {v}")
                cls.impermanents[k] = v

        for k, v in kwds.items():
            if cls.impermanents.get(k, v) != v:
                raise ValueError(f"Attempting to overwrite {k}:{cls.impermanents[k]} with {v}")
            cls.impermanents[k] = v

    key_map = {
        "delta": "δ",
        "beta": "β",
        "eta": "η",
        "epsilon": "ε",
        "zeta": "ζ",
        "ell": "ℓ",
        "repetitions": "↻",
    }
    val_map = {"beta": "%8d", "d": "%8d", "delta": "%8.6f"}

    def __init__(self, **kwds):
        for k, v in kwds.items():
            setattr(self, k, v)

    def str(self, keyword_width=None, newline=None, round_bound=2048, compact=False):  # noqa C901
        """

        :param keyword_width:  keys are printed with this width
        :param newline:        insert a newline
        :param round_bound:    values beyond this bound are represented as powers of two
        :param compact:        do not add extra whitespace to align entries

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> s = Cost(delta=5, bar=2)
            >>> s
            δ: 5.000000, bar: 2

        """

        def wfmtf(k):
            if keyword_width:
                fmt = "%%%ss" % keyword_width
            else:
                fmt = "%s"
            return fmt % k

        d = self.__dict__
        s = []
        for k, v in d.items():
            if k == "problem":  # we store the problem instance in a cost object for reference
                continue
            kk = wfmtf(self.key_map.get(k, k))
            try:
                if (1 / round_bound < abs(v) < round_bound) or (not v) or (k in self.val_map):
                    if abs(v % 1) < 0.0000001:
                        vv = self.val_map.get(k, "%8d") % round(v)
                    else:
                        vv = self.val_map.get(k, "%8.3f") % v
                else:
                    vv = "%7s" % ("≈2^%.1f" % log(v, 2))
            except TypeError:  # strings and such
                vv = "%8s" % v
            if compact:
                kk = kk.strip()
                vv = vv.strip()
            s.append(f"{kk}: {vv}")

        if not newline:
            return ", ".join(s)
        else:
            return "\n".join(s)

    def reorder(self, *args):
        """
        Return a new ordered dict from the key:value pairs in dictinonary but reordered such that the
        keys given to this function come first.

        :param args: keys which should come first (in order)

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> d = Cost(a=1,b=2,c=3); d
            a: 1, b: 2, c: 3

            >>> d.reorder("b","c","a")
            b: 2, c: 3, a: 1

        """
        keys = list(self.__dict__.keys())
        for key in args:
            keys.pop(keys.index(key))
        keys = list(args) + keys
        r = dict()
        for key in keys:
            r[key] = self.__dict__[key]
        return Cost(**r)

    def filter(self, **keys):
        """
        Return new ordered dictinonary from dictionary restricted to the keys.

        :param dictionary: input dictionary
        :param keys: keys which should be copied (ordered)
        """
        r = dict()
        for key in keys:
            r[key] = self.__dict__[key]
        return Cost(**r)

    def repeat(self, times, select=None):
        """
        Return a report with all costs multiplied by ``times``.

        :param times:  the number of times it should be run
        :param select: toggle which fields ought to be repeated and which should not
        :returns:      a new cost estimate

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> c0 = Cost(a=1, b=2)
            >>> c0.register_impermanent(a=True, b=False)
            >>> c0.repeat(1000)
            a: 1000, b: 2, ↻: 1000

        TESTS::

            >>> from estimator.cost import Cost
            >>> Cost(rop=1).repeat(1000).repeat(1000)
            rop: ≈2^19.9, ↻: ≈2^19.9

        """
        impermanents = dict(self.impermanents)

        if select is not None:
            for key in select:
                impermanents[key] = select[key]

        ret = dict()
        for key in self.__dict__:
            try:
                if impermanents[key]:
                    ret[key] = times * self.__dict__[key]
                else:
                    ret[key] = self.__dict__[key]
            except KeyError:
                raise NotImplementedError(
                    f"You found a bug, this function does not know about '{key}' but should."
                )
        ret["repetitions"] = times * ret.get("repetitions", 1)
        return Cost(**ret)

    def __rmul__(self, times):
        return self.repeat(times)

    def combine(self, right, base=None):
        """Combine ``left`` and ``right``.

        :param left: cost dictionary
        :param right: cost dictionary
        :param base: add entries to ``base``

        EXAMPLE::

            >>> from estimator.cost import Cost
            >>> c0 = Cost(a=1)
            >>> c1 = Cost(b=2)
            >>> c2 = Cost(c=3)
            >>> c0.combine(c1)
            a: 1, b: 2
            >>> c0.combine(c1, base=c2)
            c: 3, a: 1, b: 2

        """
        if base is None:
            cost = dict()
        else:
            cost = base.__dict__
        for key in self.__dict__:
            cost[key] = self.__dict__[key]
        for key in right:
            cost[key] = right.__dict__[key]
        return Cost(**cost)

    def __bool__(self):
        return self.__dict__.get("rop", oo) < oo

    def __add__(self, other):
        return self.combine(self, other)

    def __getitem__(self, key):
        return self.__dict__[key]

    def __delitem__(self, key):
        del self.__dict__[key]

    def get(self, key, default):
        return self.__dict__.get(key, default)

    def __setitem__(self, key, value):
        self.__dict__[key] = value

    def __iter__(self):
        return iter(self.__dict__)

    def values(self):
        return self.__dict__.values()

    def __repr__(self):
        return self.str(compact=True)

    def __str__(self):
        return self.str(newline=True, keyword_width=12)

    def __lt__(self, other):
        try:
            return self["rop"] < other["rop"]
        except AttributeError:
            return self["rop"] < other

    def __le__(self, other):
        try:
            return self["rop"] <= other["rop"]
        except AttributeError:
            return self["rop"] <= other
