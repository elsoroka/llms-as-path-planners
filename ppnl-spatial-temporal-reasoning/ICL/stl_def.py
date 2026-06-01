from collections.abc import Iterable
import numpy as np

# these are ''dummy'' classes used to check the generated functions are valid
class Predicate:
    def __init__(self):
        pass
    #raise NotImplementedError("Predicate is an abstract class")

class Trajectory:
    def __init__(self, z: [float]):
        self.z = np.array(z)


    def cast(self, other):
        if isinstance(other, Trajectory):
            return other
        elif isinstance(other, Iterable):
            return Trajectory(other)
        elif isinstance(other, np.ndarray):
            return Trajectory(other)
        elif isinstance(other, float) or isinstance(other, int):
            seq = [other for _ in range(len(self.z))]
            return Trajectory(seq)
        else:
            raise TypeError(f"Cannot cast {type(other)} to Trajectory")


    def __eq__(self, other):
        other = self.cast(other)
        return Equals(self, other)
    
    def __le__(self, other):
        other = self.cast(other)
        return LessThanEq(self, other)
    
    def __ge__(self, other):
        other = self.cast(other)
        return LessThanEq(other, self)
    
    def __lt__(self, other):
        other = self.cast(other)
        return LessThan(self, other)
    
    def __gt__(self, other):
        other = self.cast(other)
        return LessThan(other, self)
    

    def __hash__(self):
        return hash(tuple(self.z))

    def typecheck(self) -> None:
        pass  # base case: a Trajectory is always valid


class Equals(Predicate):
    def __init__(self, t1: Trajectory, t2: Trajectory):
        self.t1 = t1
        self.t2 = t2

    def __call__(self) -> np.array:
        return self.t1.z == self.t2.z
    
    def typecheck(self) -> None:
        if not isinstance(self.t1, Trajectory) or not isinstance(self.t2, Trajectory):
            raise TypeError("in t1 == t2, t1 and t2 must be Trajectory objects")
        
        self.t1.typecheck()
        self.t2.typecheck()


class LessThan(Predicate):
    def __init__(self, t1: Trajectory, t2: Trajectory):
        self.t1 = t1
        self.t2 = t2

    def __call__(self) -> np.array:
        return self.t1.z < self.t2.z
    
    def typecheck(self) -> None:
        if not isinstance(self.t1, Trajectory) or not isinstance(self.t2, Trajectory):
            raise TypeError("in t1 < t2, t1 and t2 must be Trajectory objects")
        
        self.t1.typecheck()
        self.t2.typecheck()


class LessThanEq(Predicate):
    def __init__(self, t1: Trajectory, t2: Trajectory):
        self.t1 = t1
        self.t2 = t2

    def __call__(self) -> np.array:
        return self.t1.z <= self.t2.z
    
    def typecheck(self) -> None:
        if not isinstance(self.t1, Trajectory) or not isinstance(self.t2, Trajectory):
            raise TypeError("in t1 <= t2, t1 and t2 must be Trajectory objects")
        
        self.t1.typecheck()
        self.t2.typecheck()



class And(Predicate):
    def __init__(self, *ps):
        if len(ps) == 1 and isinstance(ps[0], list):
            self.ps = ps[0]
        else:
            self.ps = list(ps)

    def __call__(self) -> np.array:
        if len(self.ps) == 0:
            return np.array([True])
        p1 = self.ps[0]()
        for p in self.ps[1:]:
            p1 = p1 & p() # logical and
        return p1
    
    def typecheck(self) -> None:
        for p in self.ps:
            if not isinstance(p, Predicate):
                raise TypeError("in and, all elements must be Predicate objects")
            p.typecheck()

class Or(Predicate):
    def __init__(self, *ps):
        if len(ps) == 1 and isinstance(ps[0], list):
            self.ps = ps[0]
        else:
            self.ps = list(ps)

    def __call__(self) -> np.array:
        if len(self.ps) == 0:
            return np.array([False])
        p1 = self.ps[0]()
        for p in self.ps[1:]:
            p1 = p1 | p() # logical or
        return p1
    
    def typecheck(self) -> None:
        for p in self.ps:
            if not isinstance(p, Predicate):
                raise TypeError("in or, all elements must be Predicate objects")
            p.typecheck()

class Not(Predicate):
    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def __call__(self) -> np.array:
        return ~self.predicate()
    
    def typecheck(self) -> None:
        if not isinstance(self.predicate, Predicate):
            raise TypeError("in not, predicate must be a Predicate object")
        self.predicate.typecheck()

class Implies(Predicate):
    def __init__(self, p1: Predicate, p2: Predicate):
        self.p1 = p1
        self.p2 = p2

    def __call__(self) -> np.array:
        return ~self.p1() | self.p2()
    
    def typecheck(self) -> None:
        if not isinstance(self.p1, Predicate) or not isinstance(self.p2, Predicate):
            raise TypeError("in implies, p1 and p2 must be Predicate objects")
        self.p1.typecheck()
        self.p2.typecheck()



class Always(Predicate):
    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def __call__(self) -> np.array:
        return np.array([all(self.predicate())])
    
    def typecheck(self) -> None:
        if not isinstance(self.predicate, Predicate):
            raise TypeError("in always, predicate must be a Predicate object")
        self.predicate.typecheck()

class Eventually(Predicate):
    def __init__(self, predicate: Predicate):
        self.predicate = predicate

    def __call__(self) -> np.array:
        return np.array([any(self.predicate())])
    
    def typecheck(self) -> None:
        if not isinstance(self.predicate, Predicate):
            raise TypeError("in eventually, predicate must be a Predicate object")
        self.predicate.typecheck()

class Until(Predicate):
    def __init__(self, p1: Predicate, p2: Predicate):
        self.p1 = p1
        self.p2 = p2

    def __call__(self) -> np.array:
        t1 = self.p1()
        t2 = self.p2()
        if all(t1):
            return np.array([True])
        
        for i in range(len(t1)-1):
            if all(t1[:i]) and t2[i+1]:
                return np.array([True])
            
        return np.array([False])

    
    def typecheck(self) -> None:
        if not isinstance(self.p1, Predicate) or not isinstance(self.p2, Predicate):
            raise TypeError("in until, p1 and p2 must be Predicate objects")
        self.p1.typecheck()
        self.p2.typecheck()
    

if __name__ == "__main__":
    import unittest
    
    class TestEquals(unittest.TestCase):
        def test_equals(self):
            t1 = Trajectory([1, 2, 3])
            t2 = Trajectory([1, 2, 3])
            expr = t1 == t2
            self.assertTrue(all(expr()))
            expr2 = t1 == 0
            self.assertFalse(any(expr2()))

    class TestLessThan(unittest.TestCase):
        def test_less_than(self):
            t1 = Trajectory([1, 2, 3])
            t2 = Trajectory([1, 2, 3])
            expr = t1 < t2
            self.assertFalse(any(expr()))
            expr2 = t1 < 10
            self.assertTrue(all(expr2()))

    class TestLessThanEq(unittest.TestCase):
        def test_less_than_eq(self):
            t1 = Trajectory([1, 2, 3])
            t2 = Trajectory([1, 2, 3])
            expr =t1 <= t2
            self.assertTrue(all(expr()))
            expr2 = 10 <= t1
            self.assertFalse(any(expr2()))

    class TestAnd(unittest.TestCase):
        def test_and(self):
            t1 = Trajectory([1, 2, 3])
            t2 = Trajectory([3,4,5])
            expr1 = t1 == t2
            expr2 = t1 < t2
            expr = And([expr1, expr2])
            self.assertFalse(any(expr()))
            expr2 = And([expr2, expr2])
            self.assertTrue(all(expr2()))
            expr3 = And([expr1])
            self.assertFalse(any(expr3()))

    class TestOr(unittest.TestCase):
        def test_or(self):
            t1 = Trajectory([1, 2, 3])
            t2 = Trajectory([3,4,5])
            expr1 = t1 == t2
            expr2 = t1 < t2
            expr = Or([expr1, expr2])
            self.assertTrue(all(expr()))
            expr2 = Or([expr1, expr1, expr1])
            self.assertFalse(any(expr2()))
            expr3 = Or([expr2])
            self.assertFalse(any(expr3()))

    class TestNot(unittest.TestCase):
        def test_not(self):
            t1 = Trajectory([3,4,5])
            expr1 = t1 == 0
            expr = Not(expr1)
            self.assertTrue(all(expr()))
            self.assertFalse(any(expr1()))
            self.assertFalse(any(Not(Not(expr1))()))

    class TestAlways(unittest.TestCase):
        def test_always(self):
            t1 = Trajectory([3,4,5])
            p = Equals(t1, t1)
            self.assertTrue(all(Always(p)()))
            self.assertFalse(any(Always(Not(p))()))

    class TestEventually(unittest.TestCase):
        def test_eventually(self):
            t1 = Trajectory([3,4,5])
            p = Equals(t1, t1)
            self.assertTrue(all(Eventually(p)()))
            self.assertFalse(any(Eventually(Not(p))()))

    class TestUntil(unittest.TestCase):
        def test_until(self):
            t1 = Trajectory([3,4,5])
            t2 = Trajectory([3,4,5])
            p1 = Equals(t1, t1)
            p2 = Equals(t2, t2)
            self.assertTrue(all(Until(p1, p2)()))
            self.assertTrue(all(Until(p1, Not(p2))()))

    unittest.main()