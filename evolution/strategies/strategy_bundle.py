from evolution.base.base_strategies import PopulateStrategy, FitnessStrategy, SelectionStrategy, CrossoverStrategy, \
    MutationStrategy, TerminationStrategy


class StrategyBundle:
    def __init__(self,
                 populate_strategy: PopulateStrategy,
                 fitness_strategy: FitnessStrategy,
                 selection_strategy: SelectionStrategy,
                 crossover_strategy: CrossoverStrategy,
                 mutation_strategy: MutationStrategy,
                 termination_strategy: TerminationStrategy) -> None:
        super().__init__()
        self._populate_strategy = populate_strategy
        self._fitness_strategy = fitness_strategy
        self._selection_strategy = selection_strategy
        self._crossover_strategy = crossover_strategy
        self._mutation_strategy = mutation_strategy
        self._termination_strategy = termination_strategy

    @property
    def populate_strategy(self):
        return self._populate_strategy

    @property
    def fitness_strategy(self):
        return self._fitness_strategy

    @property
    def selection_strategy(self):
        return self._selection_strategy

    @property
    def crossover_strategy(self):
        return self._crossover_strategy

    @property
    def mutation_strategy(self):
        return self._mutation_strategy

    @property
    def termination_strategy(self):
        return self._termination_strategy

    @property
    def csv(self):
        return ",".join([self._populate_strategy.printable_identifier(),
                         self._fitness_strategy.printable_identifier(),
                         self._selection_strategy.printable_identifier(),
                         self._crossover_strategy.printable_identifier(),
                         self._mutation_strategy.printable_identifier(),
                         self._termination_strategy.printable_identifier()])

    @property
    def name_identifier(self):
        return "{}_{}_{}_{}_{}_{}".format(self._populate_strategy.printable_identifier(),
                                          self._fitness_strategy.printable_identifier(),
                                          self._selection_strategy.printable_identifier(),
                                          self._crossover_strategy.printable_identifier(),
                                          self._mutation_strategy.printable_identifier(),
                                          self._termination_strategy.printable_identifier())
