def train_triplet_epoch(train_loader, model, loss_fn, optimizer, cuda, use_pair_feature=False):
    model.train()
    epoch_loss = 0
    num_examples = 0
    num_correct_triplets = 0
    if not use_pair_feature:
        for batch_idx, (data, target) in enumerate(train_loader):
            num_examples += data[0].shape[0]
            if cuda:
                data = tuple(d.float().cuda() for d in data)
            else:
                data = tuple(d.float() for d in data)

            optimizer.zero_grad()
            outputs = model(*data)

            loss, zeros = loss_fn(*outputs, size_average=False)
            num_correct_triplets += zeros
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
    else:
        for batch_idx, (node_data, edge_data, target) in enumerate(train_loader):
            num_examples += node_data[0].shape[0]
            if cuda:
                node_data = tuple(d.float().cuda() for d in node_data)
            else:
                node_data = tuple(d.float() for d in node_data)

            if cuda:
                edge_data = tuple(d.float().cuda() for d in edge_data)
            else:
                edge_data = tuple(d.float() for d in edge_data)

            optimizer.zero_grad()
            outputs = model(*node_data, *edge_data)  # TODO: check this part in details later

            loss, zeros = loss_fn(*outputs, size_average=False)
            num_correct_triplets += zeros
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

    if num_examples:
        epoch_loss /= num_examples
        non_zero_loss_triplets_ratios = 1.0 - 1.0 * num_correct_triplets / num_examples

    return epoch_loss, non_zero_loss_triplets_ratios