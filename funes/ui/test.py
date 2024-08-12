class Example:
    id_counter = 0

    @classmethod
    def builder(cls, foo):
        obj = cls()
        obj.id = cls.id_counter
        obj.foo = foo  # Set foo as an instance variable
        cls.id_counter += 1
        return obj
        example1 = Example.builder("foo_value")
        example2 = Example.builder("bar_value")