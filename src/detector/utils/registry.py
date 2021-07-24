class Registry:
    """Store classes in one place.

    Usage examples:

        >>> import torch.optim as optim
        >>> my_registry = Registry()
        >>> my_registry.add(optim.SGD)
        >>> my_registry["SGD"](*args, *kwargs)

    Add items using decorator:

        >>> my_registry = Registry()
        >>> @my_registry
        >>> class MyClass:
        >>>     pass

    """

    def __init__(self):
        self.registered_items = {}

    def add(self, item):
        """Add element to a registry.

        Args:
            item (type): class type to add to a registry
        """
        name = item.__name__
        self.registered_items[name] = item

    def __call__(self, item):
        """Add element to a registry.

        Usage examples:

            >>> class MyClass:
            >>>     pass
            >>> r = Registry()
            >>> r.add(MyClass)

        Or using as decorator:

            >>> r = Registry()
            >>> @r
            >>> class MyClass:
            >>>     pass

        Args:
            item (type): class type to add to a registry

        Returns:
            same item
        """
        self.add(item)
        return item

    def __len__(self):
        return len(self.registered_items)

    def __getitem__(self, key):
        if key not in self.registered_items:
            raise KeyError(f"Unknown or not registered key - '{key}'!")
        return self.registered_items[key]

    def __repr__(self) -> str:
        return "Registry(content={})".format(",".join(k for k in self.registered_items.keys()))
