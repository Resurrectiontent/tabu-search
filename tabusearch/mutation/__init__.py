from tabusearch.mutation.custom import create_custom_mutation

same_mutation = create_custom_mutation('Same', lambda x: [(x, '')])
