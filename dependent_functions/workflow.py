from abc import ABC, abstractmethod
import os


class Workflow(ABC):

    def __init__(self, **kwargs):
        """
        Initialize the Artifact with configuration options.

        Parameters:
        - kwargs (dict): Configuration options for the Artifact.
        """
        # Initialize properties based on kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @abstractmethod
    def read(self, filepath):
        """
        Read data from a file.

        Parameters:
        - filepath (str): Path to the file.

        Returns:
        - data: The loaded data.
        """
        pass

    @abstractmethod
    def write(self, data, filepath):
        """
        Write data to a file.

        Parameters:
        - data: The data to be written.
        - filepath (str): Path to the file.

        Returns:
        - str: The new filepath.
        """
        pass

    @abstractmethod
    def transform(self, data):
        """
        Transform the input data.

        Parameters:
        - data: The input data.

        Returns:
        - data: The transformed data.
        """
        pass

    @abstractmethod
    def is_valid_file(self, filepath):
        """
        Check if the file at the given filepath is a valid file type for this subclass.

        Parameters:
        - filepath (str): Path to the file.

        Returns:
        - bool: True if the file is a valid type, False otherwise.
        """
        pass

    def is_data_file(self, filepath):
        """
        Check if the file at the given filepath is a valid data file.

        Parameters:
        - filepath (str): Path to the file.

        Returns:
        - bool: True if the file is a valid data file, False otherwise.
        """
        # Extract the name of the Artifact class without the module information
        class_name = self.__class__.__name__

        # Check if the class name is already in the filename
        return class_name not in os.path.basename(filepath)

    def is_modified_file(self, filepath):
        """
        Check if the input file has been modified since the last processing.

        Parameters:
        - filepath (str): Path to the input file.

        Returns:
        - bool: True if the input file has been modified, False otherwise.
        """
        out_fp = self.prepare_out_path(filepath)
        if not os.path.isfile(out_fp):
            return True
        return os.path.getmtime(filepath) > os.path.getmtime(out_fp)

    def prepare_out_path(self, filepath):
        """
        Prepare the output path for the processed file, including the class name in the filename.

        Parameters:
        - filepath (str): Path to the input file.

        Returns:
        - str: The prepared output filepath.

        Raises:
        - RuntimeError: If the output filepath is the same as the input filepath.
        """
        out_path = list(filepath.split(os.path.sep))
        if out_path[-2] != "out":  # move results to "out" if not already there
            out_path.insert(-1, 'out')
        # insert a prefix, if one is provided
        try:
            out_path[-1] = f'{self.__class__.__name__}-{self.prefix}-{out_path[-1]}'
        except AttributeError:
            out_path[-1] = f'{self.__class__.__name__}-{out_path[-1]}'
        # swap extensions, if a new one is provided
        try:
            old_ext = out_path[-1].split('.')[-1]
            out_path[-1] = out_path[-1].replace(f'.{old_ext}', f'.{self.extension}')
        except AttributeError:
            pass  # "extension" is not a required attribute
        out_fp = os.path.join(*out_path)
        if out_fp == filepath:
            raise RuntimeError(f'Output filepath same as input filepath: {out_fp}')
        return out_fp

    def process(self, filepath):
        """
        Read, transform, and write data in a single call.

        Parameters:
        - filepath (str): Path to the input file.

        Returns:
        - str: The new filepath.
        """
        if not self.is_data_file(filepath):
            raise ValueError(f"Specified file is already output from {self.__class__.__name__}: {filepath}")

        if not self.is_valid_file(filepath):
            raise ValueError(f"Invalid file type for {self.__class__.__name__}: {filepath}")

        data = self.read(filepath)
        data = self.transform(data)
        # TODO: generalize to multi-input and/or multi-output cases
        out_fp = self.prepare_out_path(filepath)
        return self.write(data, out_fp)

    def __call__(self, filepath):
        """
        Make the Artifact instance callable, invoking the process method.

        Parameters:
        - filepath (str): Path to the input file.

        Returns:
        - str: The new filepath.
        """
        return self.process(filepath)


class ManyToOneWorkflow(Workflow, ABC):

    def is_data_file(self, input_files):
        return all([super().is_data_file(it) for it in input_files])

    def is_modified_file(self, input_files):
        out_fp = self.prepare_out_path(input_files)
        if not os.path.isfile(out_fp):
            return True
        return all([os.path.getmtime(it) > os.path.getmtime(out_fp) for it in input_files])

    def prepare_out_path(self, input_files):
        files_by_mtime = sorted(input_files, key=os.path.getmtime)
        filepath = files_by_mtime[-1]  # use the most recently updated file as the key
        return super().prepare_out_path(filepath)

    def process(self, input_files):
        # for filepath in input_files:
        #     if not self.is_data_file(filepath):
        #         raise ValueError(f"Specified file is already output from {self.__class__.__name__}: {filepath}")
        #
        #     if not self.is_valid_file(filepath):
        #         raise ValueError(f"Invalid file type for {self.__class__.__name__}: {filepath}")

        data = [self.read(filepath) for filepath in input_files]
        data = self.transform(data)
        out_fp = self.prepare_out_path(input_files)
        return self.write(data, out_fp)

    def __call__(self, input_files):
        return self.process(input_files)


class TwoToOneWorkflow(ManyToOneWorkflow, ABC):

    def prepare_out_path(self, input_files):
        return super().prepare_out_path(input_files[:1])
