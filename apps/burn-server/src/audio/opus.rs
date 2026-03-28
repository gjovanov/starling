//! Opus encoder for WebRTC audio streaming (ported from parakeet-rs)

use bytes::Bytes;

/// Opus encoder wrapper for WebRTC audio streaming
pub struct OpusEncoder {
    encoder: audiopus::coder::Encoder,
    resample_buffer: Vec<i16>,
    frame_size: usize,
    output_buffer: Vec<u8>,
    sequence: u16,
    timestamp: u32,
    ssrc: u32,
}

impl OpusEncoder {
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let encoder = audiopus::coder::Encoder::new(
            audiopus::SampleRate::Hz48000,
            audiopus::Channels::Mono,
            audiopus::Application::Voip,
        )?;

        let ssrc = {
            use std::time::{SystemTime, UNIX_EPOCH};
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .subsec_nanos()
        };

        Ok(Self {
            encoder,
            resample_buffer: Vec::with_capacity(960 * 2),
            frame_size: 960,
            output_buffer: vec![0u8; 1500],
            sequence: 0,
            timestamp: 0,
            ssrc,
        })
    }

    /// Encode 16kHz f32 samples to Opus RTP packets
    pub fn encode(&mut self, samples_16k: &[f32]) -> Vec<Bytes> {
        let mut packets = Vec::new();

        // Resample from 16kHz to 48kHz (3x)
        for sample in samples_16k {
            let s16 = (*sample * 32767.0) as i16;
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
            self.resample_buffer.push(s16);
        }

        // Encode complete frames
        while self.resample_buffer.len() >= self.frame_size {
            let frame: Vec<i16> = self.resample_buffer.drain(..self.frame_size).collect();

            match self.encoder.encode(&frame, &mut self.output_buffer[12..]) {
                Ok(len) => {
                    if len > 0 {
                        let packet = self.build_rtp_packet(len);
                        packets.push(packet);
                    }
                }
                Err(e) => {
                    eprintln!("[Opus] Encode error: {}", e);
                }
            }
        }

        packets
    }

    /// Build RTP packet with Opus payload
    fn build_rtp_packet(&mut self, payload_len: usize) -> Bytes {
        let mut packet = vec![0u8; 12 + payload_len];

        // RTP header
        packet[0] = 0x80; // Version 2
        packet[1] = 111; // Payload type for Opus

        // Sequence number
        packet[2] = (self.sequence >> 8) as u8;
        packet[3] = self.sequence as u8;
        self.sequence = self.sequence.wrapping_add(1);

        // Timestamp
        packet[4] = (self.timestamp >> 24) as u8;
        packet[5] = (self.timestamp >> 16) as u8;
        packet[6] = (self.timestamp >> 8) as u8;
        packet[7] = self.timestamp as u8;
        self.timestamp = self.timestamp.wrapping_add(self.frame_size as u32);

        // SSRC
        packet[8] = (self.ssrc >> 24) as u8;
        packet[9] = (self.ssrc >> 16) as u8;
        packet[10] = (self.ssrc >> 8) as u8;
        packet[11] = self.ssrc as u8;

        // Copy payload
        packet[12..12 + payload_len].copy_from_slice(&self.output_buffer[12..12 + payload_len]);

        Bytes::from(packet)
    }
}
