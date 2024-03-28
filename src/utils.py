import logging

def manipulate_passages(passages, replace_pattern, verbose=True):
    #return list(map(lambda x: x.replace(replace_pattern[0], replace_pattern[1]), passages))

    manipulated_count = 0
    manipulated_passages = []
    for passage in passages:
        if replace_pattern[0] in passage:
            passage = passage.replace(replace_pattern[0], replace_pattern[1])
            manipulated_count += 1

        manipulated_passages.append(passage)

    if verbose:
        logging.info(f"{manipulated_count} passages manipulated; '{replace_pattern[0]}' -> '{replace_pattern[1]}'")

    return manipulated_passages