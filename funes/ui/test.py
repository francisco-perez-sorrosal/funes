class Wrapper:
    def __init__(self, obj):
        self.value = obj

class MyClass:
    def __init__(self):
        self.results = []

    def my_function(self):
        for i in range(2):
            wrapper = Wrapper(i)
            yield wrapper
            self.results.append(wrapper)
        return -1

    def print_results(self):
        # for result in self.results:
        print(self.results)

# Caller code
my_object = MyClass()
generator = my_object.my_function()

# while (value := next(generator, None)) is not None:
#     print(value)

value = next(my_object.my_function(), None)
print(f"Value: {value.value}")
my_object.print_results()
value = next(my_object.my_function(), None)
print(f"Value: {value.value}")
my_object.print_results()
value = next(my_object.my_function(), None)
print(f"Value: {value}")

print(my_object.print_results())