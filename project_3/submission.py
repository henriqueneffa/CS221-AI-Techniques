from typing import List, Tuple

from mapUtil import (
    CityMap,
    computeDistance,
    createStanfordMap,
    locationFromTag,
    makeTag,
)
from util import Heuristic, SearchProblem, State, UniformCostSearch


# *IMPORTANT* :: A key part of this assignment is figuring out how to model states
# effectively. We've defined a class `State` to help you think through this, with a
# field called `memory`.
#
# As you implement the different types of search problems below, think about what
# `memory` should contain to enable efficient search!
#   > Please read the docstring for `State` in `util.py` for more details and code.

# Please also read the docstrings for the relevant classes and functions defined in `mapUtil.py`

########################################################################################
# Problem 1a: Modeling the Shortest Path Problem.


class ShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path
    from `startLocation` to any location with the specified `endTag`.
    """

    def __init__(self, startLocation: str, endTag: str, cityMap: CityMap):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        self.endLocations = [location for location, tags in cityMap.tags.items() if self.endTag in tags]

    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return State(self.startLocation) 
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return state.location in self.endLocations 
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        """
        Note we want to return a list of *3-tuples* of the form:
            (actionToReachSuccessor: str, successorState: State, cost: float)
        """
        # BEGIN_YOUR_CODE (our solution is 7 lines of code, but don't worry if you deviate from this)
        neighboringCities = self.cityMap.distances[state.location]
        result = []
        
        for succ, distance in neighboringCities.items():
            tupleForSuccessor = (succ, State(succ), distance)
            result.append(tupleForSuccessor)
            
        return result
        # END_YOUR_CODE


########################################################################################
# Problem 1b: Custom -- Plan a Route through Stanford


def getStanfordShortestPathProblem() -> ShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`endTag`. If you prefer, you may create a new map using via
    `createCustomMap()`.

    Run `python mapUtil.py > readableStanfordMap.txt` to dump a file with a list of
    locations and associated tags; you might find it useful to search for the following
    tag keys (amongst others):
        - `landmark=` - Hand-defined landmarks (from `data/stanford-landmarks.json`)
        - `amenity=`  - Various amenity types (e.g., "parking_entrance", "food")
        - `parking=`  - Assorted parking options (e.g., "underground")
    """
    cityMap = createStanfordMap()

    # Or, if you would rather use a custom map, you can uncomment the following!
    # cityMap = createCustomMap("data/custom.pbf", "data/custom-landmarks".json")

    # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
    
    #startLocation = '5714338786' # EVGR A
    #endTag = 'label=2411240427' # Coupa Green

    startLocation = '5555785613' # book store
    endTag = 'label=5648594209' # Memorial Church

    # END_YOUR_CODE
    return ShortestPathProblem(startLocation, endTag, cityMap)


########################################################################################
# Problem 2a: Modeling the Waypoints Shortest Path Problem.


class WaypointsShortestPathProblem(SearchProblem):
    """
    Defines a search problem that corresponds to finding the shortest path from
    `startLocation` to any location with the specified `endTag` such that the path also
    traverses locations that cover the set of tags in `waypointTags`.

    Hint: naively, your `memory` representation could be a list of all locations visited.
    However, that would be too large of a state space to search over! Think 
    carefully about what `memory` should represent.
    """
    def __init__(
        self, startLocation: str, waypointTags: List[str], endTag: str, cityMap: CityMap
    ):
        self.startLocation = startLocation
        self.endTag = endTag
        self.cityMap = cityMap

        # We want waypointTags to be consistent/canonical (sorted) and hashable (tuple)
        self.waypointTags = tuple(sorted(waypointTags))
        
        #ADDED BY ME:
        #precompute endlocations for the tags
        self.endLocations = [location for location, tags in cityMap.tags.items() if endTag in tags]
        self.tagToLocations = {tag: [location for location, tags in cityMap.tags.items() if tag in tags] for tag in self.waypointTags} 


    def startState(self) -> State:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        
        #start loc with way points we need to visit
        tagsGone = set(tag for tag in self.waypointTags if self.startLocation in self.tagToLocations.get(tag, []))
        tagsLeft = tuple(sorted(set(self.waypointTags) - tagsGone))
        return State(self.startLocation,tagsLeft)
        
        # END_YOUR_CODE

    def isEnd(self, state: State) -> bool:
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        
        return state.location in (self.endLocations) and len(state.memory) <= 0
        
        # END_YOUR_CODE

    def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
        # BEGIN_YOUR_CODE (our solution is 17 lines of code, but don't worry if you deviate from this)
        
        # return successors/cost for the state. Removes tag if location matches waypoint
        neighboringCities = self.cityMap.distances[state.location]
        result = []
        
        for succ, distance in neighboringCities.items():
            # Removes tag if location matches waypoint

            tagsGone = set(tag for tag in state.memory if succ in self.tagToLocations.get(tag, []))
            tagsLeft = tuple(sorted(set(state.memory) - tagsGone))
            
            tupleForSuccessor = (succ, State(succ, tagsLeft), distance)
            result.append(tupleForSuccessor)
            
        return result
        
        # END_YOUR_CODE


########################################################################################
# Problem 2b: Custom -- Plan a Route with Unordered Waypoints through Stanford


def getStanfordWaypointsShortestPathProblem() -> WaypointsShortestPathProblem:
    """
    Create your own search problem using the map of Stanford, specifying your own
    `startLocation`/`waypointTags`/`endTag`.

    Similar to Problem 1b, use `readableStanfordMap.txt` to identify potential
    locations and tags.
    """
    cityMap = createStanfordMap()
    # BEGIN_YOUR_CODE (our solution is 4 lines of code, but don't worry if you deviate from this)
    
    #startLocation = '5555785613'  # book store
    startLocation = '5714338786' # EVGR A

    
    waypointTags = ['amenity=food', 'amenity=parking_entrance']
    
    endTag = 'label=5648594209'  # Memorial Church
    #endTag = 'label=2411240427' # Coupa Green


    # END_YOUR_CODE
    return WaypointsShortestPathProblem(startLocation, waypointTags, endTag, cityMap)


########################################################################################
# Problem 3a: A* to UCS reduction

# Turn an existing SearchProblem (`problem`) you are trying to solve with a
# Heuristic (`heuristic`) into a new SearchProblem (`newSearchProblem`), such
# that running uniform cost search on `newSearchProblem` is equivalent to
# running A* on `problem` subject to `heuristic`.
#
# This process of translating a model of a problem + extra constraints into a
# new instance of the same problem is called a reduction; it's a powerful tool
# for writing down "new" models in a language we're already familiar with.


def aStarReduction(problem: SearchProblem, heuristic: Heuristic) -> SearchProblem:
    class NewSearchProblem(SearchProblem):
        def startState(self) -> State:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
            
            return problem.startState()
            
            # END_YOUR_CODE

        def isEnd(self, state: State) -> bool:
            # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
             
            return problem.isEnd(state)
            
            # END_YOUR_CODE

        def successorsAndCosts(self, state: State) -> List[Tuple[str, State, float]]:
            # BEGIN_YOUR_CODE (our solution is 8 lines of code, but don't worry if you deviate from this)
            # adjust the cost cost(original) + heuristic(successor) - heuristic(state)
            successors = problem.successorsAndCosts(state)
            newSuccessors=[]
            
            for action,newState,cost in successors:
                #adjustment of the cost as decribed
                costAdj = cost+heuristic.evaluate(newState)-heuristic.evaluate(state)
                newSuccessors.append((action, newState, costAdj))
            
            return newSuccessors

            # END_YOUR_CODE

    return NewSearchProblem()


########################################################################################
# Problem 3b: "straight-line" heuristic for A*


class StraightLineHeuristic(Heuristic):
    """
    Estimate the cost between locations as the straight-line distance.
        > Hint: you might consider using `computeDistance` defined in `mapUtil.py`
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        self.endTag = endTag
        self.cityMap = cityMap

        # Precompute
        # BEGIN_YOUR_CODE (our solution is 5 lines of code, but don't worry if you deviate from this)
        
        self.endLocations = [location for location, tags in cityMap.tags.items() if endTag in tags]
        # get geo locations
        self.endGeos = [self.cityMap.geoLocations[loc] for loc in self.endLocations] 
        
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 6 lines of code, but don't worry if you deviate from this)
        
        #get geo for current state
        geo = self.cityMap.geoLocations[state.location]
        #straight line distance calculation
        distances = [computeDistance(geo, g) for g in self.endGeos]
        return min(distances)
        
        # END_YOUR_CODE

########################################################################################
# Problem 3c: "no waypoints" heuristic for A*


class NoWaypointsHeuristic(Heuristic):
    """
    Returns the minimum distance from `startLocation` to any location with `endTag`,
    ignoring all waypoints.
    """
    def __init__(self, endTag: str, cityMap: CityMap):
        """
        Precompute cost of shortest path from each location to a location with the desired endTag
        """
        #added
        endLocations = [location for location, tags in cityMap.tags.items() if endTag in tags]

        # Define a reversed shortest path problem from a special END state
        # (which connects via 0 cost to all end locations) to `startLocation`.
        class ReverseShortestPathProblem(SearchProblem):
            def startState(self) -> State:
                """
                Return special "END" state
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return State("END")
                # END_YOUR_CODE

            def isEnd(self, state: State) -> bool:
                """
                Return False for each state.
                Because there is *not* a valid end state (`isEnd` always returns False), 
                UCS will exhaustively compute costs to *all* other states.
                """
                # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
                return False
                # END_YOUR_CODE

            def successorsAndCosts(
                self, state: State
            ) -> List[Tuple[str, State, float]]:
                # If current location is the special "END" state, 
                # return all the locations with the desired endTag and cost 0 
                # (i.e, we connect the special location "END" with cost 0 to all locations with endTag)
                # Else, return all the successors of current location and their corresponding distances according to the cityMap
                # BEGIN_YOUR_CODE (our solution is 14 lines of code, but don't worry if you deviate from this)
                if state.location == "END":
                
                    return [(endLoc, State(endLoc), 0) for endLoc in endLocations]
                
                else:
                    return [
                        
                        (succ, State(succ), distance)

                        for succ, distance in cityMap.distances[state.location].items() 
                    ]
                
                # END_YOUR_CODE

        # Call UCS.solve on our `ReverseShortestPathProblem` instance. Because there is
        # *not* a valid end state (`isEnd` always returns False), will exhaustively
        # compute costs to *all* other states.
        # BEGIN_YOUR_CODE (our solution is 2 lines of code, but don't worry if you deviate from this)
        
        ucs = UniformCostSearch()
        
        ucs.solve(ReverseShortestPathProblem())
        
        # END_YOUR_CODE

        # Now that we've exhaustively computed costs from any valid "end" location
        # (any location with `endTag`), we can retrieve `ucs.pastCosts`; this stores
        # the minimum cost path to each state in our state space.
        #   > Note that we're making a critical assumption here: costs are symmetric!
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        self.pastCosts = ucs.pastCosts
        # END_YOUR_CODE

    def evaluate(self, state: State) -> float:
        # BEGIN_YOUR_CODE (our solution is 1 line of code, but don't worry if you deviate from this)
        return self.pastCosts[state.location]
        # END_YOUR_CODE
