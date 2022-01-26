class Singleton(type):
    """
    class MySingleton(metaclass=Singleton):
        ...

    Setting `metaclass=Singleton` in the classes meta descriptor marks it as a
    singleton object: if the object has already been constructed elsewhere in
    the code, subsequent calls to the constructor just return this original
    instance.
    """

    # Stores instances in a dictionary:
    # {class: instance}
    _instances = dict()

    def __call__(cls, *args, **kwargs):
        """
        Metclass __call__ operator is called before the class constructor -- so
        this operator will check if an instance already exists in
        Singleton._instances. If it doesn't call the constructor and add the
        instance to Singleton._instances. If it does, then don't call the
        constructor and return the instance instead.
        """
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(
                *args, **kwargs
            )

        return cls._instances[cls]