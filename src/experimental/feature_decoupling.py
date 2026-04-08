import torch
import torch.nn as nn


class AdaINLipReadingModel(nn.Module):
    def __init__(self, vocab_size):
        super(AdaINLipReadingModel, self).__init__()
        self.personality_extractor = PersonalityFeatureExtractor()

        # 3DCNN + ResNet + Conformer encoder
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Add more layers as needed
        )

        # Transformer decoder
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8),
            num_layers=6
        )

        self.fc_out = nn.Linear(512, vocab_size)

        # Neutral style parameters
        self.register_buffer('neutral_mean', torch.zeros(128))
        self.register_buffer('neutral_std', torch.ones(128))

    def adain(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = style_feat.mean(dim=[2, 3]), style_feat.std(dim=[2, 3])
        normalized_feat = (content_feat - content_feat.mean(dim=[2, 3]).unsqueeze(-1).unsqueeze(-1)) / content_feat.std(
            dim=[2, 3]).unsqueeze(-1).unsqueeze(-1)
        return normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1) + style_mean.unsqueeze(-1).unsqueeze(-1)

    def update_neutral_style(self, static_features, dynamic_features):
        # Update neutral style parameters
        self.neutral_mean = 0.9 * self.neutral_mean + 0.1 * torch.mean(static_features, dim=0)
        self.neutral_std = 0.9 * self.neutral_std + 0.1 * torch.std(static_features, dim=0)

    def forward(self, video_input, static_input, dynamic_input):
        static_features, dynamic_features = self.personality_extractor(static_input, dynamic_input)

        # Update neutral style
        self.update_neutral_style(static_features, dynamic_features)

        # Encode video
        encoded = self.encoder(video_input)

        # Apply AdaIN
        personality_adjusted = self.adain(encoded, static_features.unsqueeze(-1).unsqueeze(-1))

        # Interpolate between neutral and personal style
        alpha = 0.5  # Can be adjusted or learned
        interpolated = alpha * personality_adjusted + (1 - alpha) * encoded

        # Decode
        decoded = self.decoder(interpolated)

        # Output layer
        output = self.fc_out(decoded)

        return output


# Usage example
vocab_size = 1000  # Example vocabulary size
model = AdaINLipReadingModel(vocab_size)

# Dummy inputs
video_input = torch.randn(1, 3, 16, 112, 112)  # Batch, Channels, Time, Height, Width
static_input = torch.randn(1, 3, 112, 112)  # Single frame for static features
dynamic_input = compute_optical_flow(video_input.squeeze(0))  # Optical flow

output = model(video_input, static_input, dynamic_input)
print(output.shape)  # Should be [1, 16, 1000] (Batch, Time, Vocab_size)







import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np


class AdaINLipReadingModel(nn.Module):
    def __init__(self, vocab_size, num_speakers):
        super(AdaINLipReadingModel, self).__init__()
        self.personality_extractor = PersonalityFeatureExtractor(num_speakers)

        # 3DCNN + ResNet + Conformer encoder (simplified for brevity)
        self.encoder = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2)),
            # Add more layers as needed
        )

        # Transformer decoder (simplified for brevity)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=512, nhead=8),
            num_layers=6
        )

        self.fc_out = nn.Linear(512, vocab_size)

        # Neutral style parameters
        self.register_buffer('neutral_mean', torch.zeros(128))
        self.register_buffer('neutral_std', torch.ones(128))

        # Learnable interpolation parameter
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def adain(self, content_feat, style_feat):
        size = content_feat.size()
        style_mean, style_std = style_feat.mean([2, 3]), style_feat.std([2, 3])
        normalized_feat = (content_feat - content_feat.mean([2, 3]).unsqueeze(-1).unsqueeze(-1)) / (
                    content_feat.std([2, 3]).unsqueeze(-1).unsqueeze(-1) + 1e-8)
        return normalized_feat * style_std.unsqueeze(-1).unsqueeze(-1) + style_mean.unsqueeze(-1).unsqueeze(-1)

    def update_neutral_style(self, static_features, dynamic_features):
        # Update neutral style parameters
        self.neutral_mean = 0.9 * self.neutral_mean + 0.1 * torch.mean(static_features, dim=0)
        self.neutral_std = 0.9 * self.neutral_std + 0.1 * torch.std(static_features, dim=0)

    def forward(self, video_input, static_input, dynamic_input):
        static_features, dynamic_features, _, _ = self.personality_extractor(static_input, dynamic_input)

        # Update neutral style
        self.update_neutral_style(static_features, dynamic_features)

        # Encode video
        encoded = self.encoder(video_input)

        # Apply AdaIN
        personality_adjusted = self.adain(encoded, static_features.unsqueeze(-1).unsqueeze(-1))

        # Interpolate between neutral and personal style
        interpolated = self.alpha * personality_adjusted + (1 - self.alpha) * self.adain(encoded,
                                                                                         self.neutral_mean.unsqueeze(
                                                                                             -1).unsqueeze(-1))

        # Decode
        decoded = self.decoder(interpolated)

        # Output layer
        output = self.fc_out(decoded)

        return output, static_features, dynamic_features


def compute_lip_reading_loss(pred, target):
    # Implement your lip reading loss here
    # For example, you could use CrossEntropyLoss if pred is logits and target is class indices
    return nn.CrossEntropyLoss()(pred, target)


def compute_content_loss(features1, features2):
    return nn.MSELoss()(features1, features2)


def train_decoupling(model, train_loader, lip_reading_criterion, optimizer, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (video_input, static_input, dynamic_input, lip_reading_target) in enumerate(train_loader):
            video_input, static_input, dynamic_input, lip_reading_target = \
                video_input.to(device), static_input.to(device), dynamic_input.to(device), lip_reading_target.to(device)

            optimizer.zero_grad()

            # Forward pass
            lip_reading_output, static_features, dynamic_features = model(video_input, static_input, dynamic_input)

            # Compute losses
            lip_reading_loss = lip_reading_criterion(lip_reading_output, lip_reading_target)
            content_loss = compute_content_loss(static_features, dynamic_features)

            # Total loss
            total_loss = lip_reading_loss + 0.1 * content_loss

            # Backward pass and optimize
            total_loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f'Epoch {epoch + 1}, Batch {batch_idx}, Loss: {total_loss.item():.4f}')


def main():
    # Hyperparameters
    batch_size = 32
    num_epochs = 20
    learning_rate = 0.001
    vocab_size = 1000  # Vocabulary size for lip reading
    num_speakers = 100  # Number of unique speakers in the dataset

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = AdaINLipReadingModel(vocab_size, num_speakers).to(device)

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create data loader
    # Note: You need to implement your own dataset class that returns (video_input, static_input, dynamic_input, lip_reading_target)
    train_loader = DataLoader(your_dataset, batch_size=batch_size, shuffle=True)

    # Train the model
    train_decoupling(model, train_loader, compute_lip_reading_loss, optimizer, device, num_epochs)

    print('Training finished')


if __name__ == '__main__':
    main()