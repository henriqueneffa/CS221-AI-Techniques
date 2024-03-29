import collections
import os
import sys
from typing import List, Tuple

from logic import *

############################################################
# Problem 1: propositional logic
# Convert each of the following natural language sentences into a propositional
# logic formula.  See rainWet() in examples.py for a relevant example.

# Sentence: "If it's summer and we're in California, then it doesn't rain."
def formula1a() -> Formula:
    # Predicates to use:
    Summer = Atom('Summer')               # whether it's summer
    California = Atom('California')       # whether we're in California
    Rain = Atom('Rain')                   # whether it's raining
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    # Implies for conditional logic; And to combine conditions; Not for negation. Implication
    return Implies(And(Summer,California), Not(Rain)) 
    
    # END_YOUR_CODE

# Sentence: "It's wet if and only if it is raining or the sprinklers are on."
def formula1b() -> Formula:
    # Predicates to use:
    Rain = Atom('Rain')              # whether it is raining
    Wet = Atom('Wet')                # whether it it wet
    Sprinklers = Atom('Sprinklers')  # whether the sprinklers are on
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    # Equiv for equivalence relation; Or for disjunction. If and only if + Or 
    return Equiv(Wet, Or(Rain,Sprinklers))
    
    # END_YOUR_CODE

# Sentence: "Either it's day or night (but not both)."
def formula1c() -> Formula:
    # Predicates to use:
    Day = Atom('Day')     # whether it's day
    Night = Atom('Night') # whether it's night
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    #  'And'; 'Or'; 'Not' for negating conjunction. Day or night or not both...
    return And(Or(Day, Night), Not(And(Day, Night)))
    
    # END_YOUR_CODE

############################################################
# Problem 2: first-order logic

# Sentence: "Every person has a parent."
def formula2a() -> Formula:
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Parent(x, y): return Atom('Parent', x, y)  # whether x's parent is y

    # Note: You do NOT have to enforce that the parent is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    # for all individuals x ( if x is a person (Person('$x'))  ) 
    # then there exists an individual y who is a parent of x
    return Forall('$x', Implies(Person('$x'), Exists('$y',Parent('$x','$y')))) 
    
    # END_YOUR_CODE

# Sentence: "At least one person has no children."
def formula2b() -> Formula:
    # Predicates to use:
    def Person(x): return Atom('Person', x)        # whether x is a person
    def Child(x, y): return Atom('Child', x, y)    # whether x has a child y

    # Note: You do NOT have to enforce that the child is a "person"
    # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
    
    # look for one individual (Exists) that is a person (Person) AND Person x has no children
    return Exists('$x', And(Person('$x'),Forall('$y',Not(Child('$x','$y')))))
    
    # END_YOUR_CODE

# Return a formula which defines Father in terms of Male and Parent
# See parentChild() in examples.py for a relevant example.
def formula2c() -> Formula:
    # Predicates to use:
    def Male(x): return Atom('Male', x)                  # whether x is male
    def Parent(x, y): return Atom('Parent', x, y)        # whether x has a parent y
    def Father(x, y): return Atom('Father', x, y)        # whether x has a father y
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    # x is the father of y if and only if y is male and x is a parent of x
    # Forall iterates on all pais; Equivlook if matches between Father and Male + Parent
    return Forall('$y', Forall('$x', Equiv(Father('$x','$y'), And(Male('$y'), Parent('$x','$y')))))
    
    # END_YOUR_CODE

# Return a formula which defines Granddaughter in terms of Female and Child.
# Note: It is ok for a person to be her own child
def formula2d() -> Formula:
    # Predicates to use:
    def Female(x): return Atom('Female', x)                      # whether x is female
    def Child(x, y): return Atom('Child', x, y)                  # whether x has a child y
    def Granddaughter(x, y): return Atom('Granddaughter', x, y)  # whether x has a graddaughter y
    # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
    
    # for a pair (granddaughter & grandparent)
    # check if  granddaughter is female and if parent's who
    # is the child of the grandparent and also the parent of  granddaughter
    return Forall('$person', Forall('$grandparent', 
                Equiv(Granddaughter('$grandparent', '$person'), 
                And(
                    Female('$person'), 
                    Exists('$parent', And(Child('$parent', '$person'), Child('$grandparent', '$parent')))))))
    
    # END_YOUR_CODE

############################################################
# Problem 3: Liar puzzle

# Facts:
# 0. Mark: "It wasn't me!"
# 1. John: "It was Nicole!"
# 2. Nicole: "No, it was Susan!"
# 3. Susan: "Nicole's a liar."
# 4. Exactly one person is telling the truth.
# 5. Exactly one person crashed the server.
# Query: Who did it?
# This function returns a list of 6 formulas corresponding to each of the
# above facts. Be sure your formulas are exactly in the order specified. 
# Hint: You might want to use the Equals predicate, defined in logic.py.  This
# predicate is used to assert that two objects are the same.
# In particular, Equals(x,x) = True and Equals(x,y) = False iff x is not equal to y.
def liar() -> Tuple[List[Formula], Formula]:
    def TellTruth(x): return Atom('TellTruth', x)
    def CrashedServer(x): return Atom('CrashedServer', x)
    mark = Constant('mark')
    john = Constant('john')
    nicole = Constant('nicole')
    susan = Constant('susan')
    formulas = []
    # We provide the formula for fact 0 here.
    formulas.append(Equiv(TellTruth(mark), Not(CrashedServer(mark))))
    # You should add 5 formulas, one for each of facts 1-5.
    # BEGIN_YOUR_CODE (our solution is 11 lines of code, but don't worry if you deviate from this)



    formulas.append(Equiv(TellTruth(john), CrashedServer(nicole)))
    formulas.append(Equiv(TellTruth(nicole), CrashedServer(susan)))
    formulas.append(Equiv(TellTruth(susan), Not(TellTruth(nicole))))

    formulas.append(Exists('$x', And(TellTruth('$x'), Forall('$y', Implies(Not(Equals('$x', '$y')), Not(TellTruth('$y')))))))
    formulas.append(Exists('$x', And(CrashedServer('$x'), Forall('$y', Implies( Not(Equals('$x', '$y')), Not(CrashedServer('$y')))))))



    # END_YOUR_CODE
    query = CrashedServer('$x')
    return (formulas, query)

############################################################
# Problem 4: Odd and even integers

# Return the following 6 laws. Be sure your formulas are exactly in the order specified.
# 0. Each number $x$ has exactly one successor, which is not equal to $x$.
# 1. Each number is either even or odd, but not both.
# 2. The successor number of an even number is odd.
# 3. The successor number of an odd number is even.
# 4. For every number $x$, the successor of $x$ is larger than $x$.
# 5. Larger is a transitive property: if $x$ is larger than $y$ and $y$ is
#    larger than $z$, then $x$ is larger than $z$.
# Query: For each number, there exists an even number larger than it.
def ints() -> Tuple[List[Formula], Formula]:
    def Even(x): return Atom('Even', x)                  # whether x is even
    def Odd(x): return Atom('Odd', x)                    # whether x is odd
    def Successor(x, y): return Atom('Successor', x, y)  # whether x's successor is y
    def Larger(x, y): return Atom('Larger', x, y)        # whether x is larger than y
    # Note: all objects are numbers, so we don't need to define Number as an
    # explicit predicate.
    # Note: pay attention to the order of arguments of Successor and Larger.
    # Populate |formulas| with the 6 laws above and set |query| to be the
    # query.
    # Hint: You might want to use the Equals predicate, defined in logic.py.  This
    # predicate is used to assert that two objects are the same.
    formulas = []
    query = None
    # BEGIN_YOUR_CODE (our solution is 23 lines of code, but don't worry if you deviate from this)
    x = '$x'
    y = '$y'
    z = '$z'
    formulas.append(
        Forall(x,
               Exists(y, And(
                    Not(Equals(x, y)), 
                    And(Successor(x, y), Forall(z, Implies(Not(Equals(z, y)), Not(Successor(x, z)))))
                ))
    ))
    formulas.append(Forall(x, Xor(Even(x), Odd(x))))
    formulas.append(Forall(x, Forall(y, Implies(And(Even(x), Successor(x, y)), Odd(y)))))
    formulas.append(Forall(x, Forall(y, Implies(And(Odd(x), Successor(x, y)), Even(y)))))
    formulas.append(Forall(x, Forall(y, Implies(Successor(x, y), Larger(y, x)))))
    formulas.append(Forall(x, Forall(y, Forall(z, Implies(And(Larger(x, y), Larger(y, z)), Larger(x, z))))))
    # END_YOUR_CODE
    query = Forall('$x', Exists('$y', And(Even('$y'), Larger('$y', '$x'))))
    return (formulas, query)

############################################################
# Problem 5: semantic parsing
# Each of the following functions should return a GrammarRule.
# Look at createBaseEnglishGrammar() in nlparser.py to see what these rules should look like.
# For example, the rule for 'X is a Y' is:
#     GrammarRule('$Clause', ['$Name', 'is', 'a', '$Noun'],
#                 lambda args: Atom(args[1].title(), args[0].lower()))
# Note: args[0] corresponds to $Name and args[1] corresponds to $Noun.
# Note: by convention, .title() should be applied to all predicates (e.g., Cat).
# Note: by convention, .lower() should be applied to constant symbols (e.g., garfield).

from nlparser import GrammarRule


def createRule1() -> GrammarRule:
    # Return a GrammarRule for 'every $Noun $Verb some $Noun'
    # Note: universal quantification should be outside existential quantification.
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

def createRule2() -> GrammarRule:
    # Return a GrammarRule for 'there is some $Noun that every $Noun $Verb'
    # BEGIN_YOUR_CODE (our solution is 3 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE

def createRule3() -> GrammarRule:
    # Return a GrammarRule for 'if a $Noun $Verb a $Noun then the former $Verb the latter'
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    raise Exception("Not implemented yet")
    # END_YOUR_CODE
